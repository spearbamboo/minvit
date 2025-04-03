# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-supervised training script for low resolution small-scale datasets using DataParallel (DP)"""

import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from utils import utils_ssl as utils
from projection_head import MLPHead
from functools import partial
from models.vit import VisionTransformer
from models.swin import SwinTransformer
from models.cait import cait_models
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('SSL for low resolution dataset', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit', type=str,
        choices=['vit', 'swin', 'cait'] + torchvision_archs,
        help="""Name of architecture to train. For quick experiments with ViTs, we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=4, type=int, help="""Size in pixels of input square patches - default 4 (for 4x4 patches) """)
    parser.add_argument('--out_dim', default=1024, type=int, help="""Dimensionality of the SSL MLP head output. For complex and large datasets large values (like 65k) work well.""")

    parser.add_argument('--norm_last_layer', default=False, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the MLP head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    parser.add_argument('--image_size', default=32, type=int, help="""Size of input image.""")
    parser.add_argument('--in_channels', default=3, type=int, help="""Input image channels.""")
    parser.add_argument('--embed_dim', default=192, type=int, help="""Dimensions of ViT.""")
    parser.add_argument('--num_layers', default=9, type=int, help="""Number of layers of ViT.""")
    parser.add_argument('--num_heads', default=12, type=int, help="""Number of heads in attention layer in ViT.""")
    parser.add_argument('--vit_mlp_ratio', default=2, type=int, help="""MLP hidden dimension ratio.""")
    parser.add_argument('--qkv_bias', default=True, type=bool, help="""Bias in Q, K, and V values.""")
    parser.add_argument('--drop_rate', default=0., type=float, help="""Dropout rate.""")
    parser.add_argument('--vit_init_value', default=0.1, type=float, help="""Initialisation values of ViT.""")
    parser.add_argument('--use_ape', default=False, type=bool, help="""Absolute position embeddings.""")
    parser.add_argument('--use_rpb', default=False, type=bool, help="""Relative position embeddings.""")
    parser.add_argument('--use_shared_rpb', default=False, type=bool, help="""Shared Relative position embeddings.""")
    parser.add_argument('--use_mean_pooling', default=False, type=bool, help="""Use mean pooling instead of CLS token.""")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.07, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increasing this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=10, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 10).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the weight decay.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the weight decay.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter gradient norm if using gradient clipping.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Batch size per GPU.')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs during which the output layer is frozen.""")
    parser.add_argument("--lr", default=0.0001, type=float, help="""Learning rate at the end of warmup.""")
    parser.add_argument("--warmup_epochs", default=30, type=int,
        help="Number of epochs for linear learning-rate warmup.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target learning rate at the end of training.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. adamw is recommended with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="Stochastic depth rate.")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.5, 1.),
        help="""Scale range for global crops.""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of local crops to generate.""")
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.2, 0.4),
        help="""Scale range for local crops.""")
    
    # Miscellaneous
    parser.add_argument('--dataset', default='CIFAR10', type=str,
        choices=['Tiny-Imagenet', 'CIFAR10', 'CIFAR100', 'CINIC', 'SVHN'],
        help='Dataset name.')
    parser.add_argument('--datapath', default='./data', type=str,
        help='Path to the training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=10, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers.')
    # Distributed parameters are not used in DP mode, so these can be ignored.
    parser.add_argument("--dist_url", default="env://", type=str, help="URL for distributed training (ignored in DP mode).")
    parser.add_argument("--local_rank", default=0, type=int, help="Local rank (ignored in DP mode).")
    parser.add_argument("--mlp_head_in", default=192, type=int, help="Input dimension for the MLP projection head.")
    return parser

def train(args):
    # For DataParallel mode, we do not initialize distributed training.
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ============
    transform = DataAugmentation(args)
    if args.dataset == 'Tiny-Imagenet':
        dataset = datasets.ImageFolder(root=args.datapath, transform=transform)
    elif args.dataset == "CIFAR10":
        dataset = torchvision.datasets.CIFAR10(root=args.datapath, train=True,
                                       download=True, transform=transform)
    elif args.dataset == "CIFAR100":
        dataset = torchvision.datasets.CIFAR100(root=args.datapath, train=True,
                                       download=True, transform=transform)
    elif args.dataset == "CINIC":
        dataset = datasets.ImageFolder(root=args.datapath, transform=transform)
    elif args.dataset == "SVHN":
        dataset = torchvision.datasets.SVHN(root=args.datapath, split='train',
                                         download=True, transform=transform)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ============
    if args.arch == 'vit':
        student = VisionTransformer(
            img_size=[args.image_size],
            patch_size=args.patch_size,
            in_chans=args.in_channels,
            num_classes=0,
            embed_dim=192,
            depth=9,
            num_heads=12,
            mlp_ratio=2,
            qkv_bias=args.qkv_bias,
            drop_rate=args.drop_rate,
            drop_path_rate=args.drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        teacher = VisionTransformer(
            img_size=[args.image_size],
            patch_size=args.patch_size,
            in_chans=args.in_channels,
            num_classes=0,
            embed_dim=192,
            depth=9,
            num_heads=12,
            mlp_ratio=2,
            qkv_bias=args.qkv_bias,
            drop_rate=args.drop_rate,
            drop_path_rate=args.drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
    elif args.arch == 'swin':
        mlp_ratio = args.vit_mlp_ratio
        window_size = 4
        patch_size = 2 if args.image_size == 32 else 4

        student = SwinTransformer(
            img_size=args.image_size,
            num_classes=0,
            window_size=window_size,
            patch_size=patch_size,
            embed_dim=96,
            depths=[2, 6, 4],
            num_heads=[3, 6, 12],
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop_path_rate=args.drop_path_rate
        )
        teacher = SwinTransformer(
            img_size=args.image_size,
            num_classes=0,
            window_size=window_size,
            patch_size=patch_size,
            embed_dim=96,
            depths=[2, 6, 4],
            num_heads=[3, 6, 12],
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop_path_rate=args.drop_path_rate
        )
    elif args.arch == 'cait':
        patch_size = 4 if args.image_size == 32 else 8
        student = cait_models(
            img_size=args.image_size,
            patch_size=patch_size,
            embed_dim=192,
            depth=24,
            num_heads=4,
            mlp_ratio=2,
            qkv_bias=True,
            num_classes=0,
            drop_path_rate=args.drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_scale=1e-5,
            depth_token_only=2
        )
        teacher = cait_models(
            img_size=args.image_size,
            patch_size=patch_size,
            embed_dim=192,
            depth=24,
            num_heads=4,
            mlp_ratio=2,
            qkv_bias=True,
            num_classes=0,
            drop_path_rate=args.drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_scale=1e-5,
            depth_token_only=2
        )
    else:
        print(f"Unknown architecture: {args.arch}")
        sys.exit(1)

    # Wrap student and teacher with MultiCropWrapper and MLPHead
    student = utils.MultiCropWrapper(student, MLPHead(args.mlp_head_in, args.out_dim, args.use_bn_in_head))
    teacher = utils.MultiCropWrapper(
        teacher, MLPHead(
            args.mlp_head_in,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        )
    )

    # ============ moving networks to GPU and wrapping with DataParallel ============
    student, teacher = student.cuda(), teacher.cuda()
    teacher = nn.DataParallel(teacher)
    # teacher_without_dp refers to the underlying module (needed for weight sync and updates)
    teacher_without_dp = teacher.module
    student = nn.DataParallel(student)

    # teacher and student start with the same weights
    filtered_student_state_dict = {k: v for k, v in student.module.state_dict().items() if k in teacher_without_dp.state_dict().keys()}
    teacher_without_dp.load_state_dict(filtered_student_state_dict)

    # Disable gradient computation for teacher
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"Student and Teacher are built: they are both {args.arch} networks.")

    # ============ preparing loss ============
    view_pred_loss = ViewPredLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local crops
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * torch.cuda.device_count()) / 256.,
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # Optionally resume training
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        view_pred_loss=view_pred_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting SSL training!")
    for epoch in range(start_epoch, args.epochs):
        # In DP mode, no need to set epoch for sampler
        train_stats = train_one_epoch(student, teacher, teacher_without_dp, view_pred_loss,
                                      data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                                      epoch, fp16_scaler, args)

        # ============ saving checkpoints ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'view_pred_loss': view_pred_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_dp, view_pred_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        it_global = len(data_loader) * epoch + it  # global training iteration
        # update learning rate and weight decay for each param group
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it_global]
            if i == 0:
                param_group["weight_decay"] = wd_schedule[it_global]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]

        # forward passes of teacher and student and compute loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only global crops for teacher
            student_output = student(images)
            total_loss_dict = view_pred_loss(student_output, teacher_output, epoch)
            loss = total_loss_dict.pop('loss')
            loss_view = total_loss_dict.pop('ce_loss')

        if not math.isfinite(loss.item()):
            print("Loss is {}, View Pred loss is {}, stopping training".format(loss.item(), loss_view.item()), force=True)
            sys.exit(1)

        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for teacher
        with torch.no_grad():
            m = momentum_schedule[it_global]
            names_q, params_q, names_k, params_k = [], [], [], []
            for name_q, param_q in student.module.named_parameters():
                names_q.append(name_q)
                params_q.append(param_q)
            for name_k, param_k in teacher_without_dp.named_parameters():
                names_k.append(name_k)
                params_k.append(param_k)
            names_common = list(set(names_q) & set(names_k))
            params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
            params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]
            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(view_pred_loss=loss_view.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class ViewPredLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, in_channels=3, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.in_channels = in_channels
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # Teacher output: apply centering and temperature sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        n_loss_terms = 0
        total_loss = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        loss_dict = dict(ce_loss=total_loss, loss=total_loss)
        self.update_center(teacher_output)
        return loss_dict

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # In DataParallel mode, no distributed reduction is required.
        batch_center = batch_center / len(teacher_output)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentation(object):
    def __init__(self, args):
        if args.dataset == "CIFAR10":
            mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        elif args.dataset == "CIFAR100":
            mean, std = (0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        elif args.dataset == "SVHN":
            mean, std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
        elif args.dataset == "CINIC":
            mean, std = (0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835)
        elif args.dataset == "Tiny-Imagenet":
            mean, std = (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)

        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size, scale=args.global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size, scale=args.global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        self.local_crops_number = args.local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size // 2, scale=args.local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SSL for low resolution dataset', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)
