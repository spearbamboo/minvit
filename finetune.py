"""
CUDA_VISIBLE_DEVICES=0 wandb run --job_type=finetune finetune.py --datapath ~/gent/data --batch_size=256 --amp --pretrained_weights /path/to/train_ssl_checkpoint.pth
"""

import wandb
from utils.mix import cutmix_data, mixup_data, mixup_criterion
import numpy as np
import random
import logging as log
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from colorama import Fore, Style
# torch.distributed 관련 코드는 제거되었습니다.
from utils.losses import LabelSmoothingCrossEntropy
import os
from utils.sampler import RASampler
from utils.logger_dict import Logger_dict
from utils.print_progress import progress_bar
from utils.training_functions import accuracy
import argparse
from utils.scheduler import build_scheduler
from utils.dataloader import datainfo, dataload
from models.build_model import create_model
from tqdm import tqdm
import warnings
import gin
warnings.filterwarnings("ignore", category=Warning)

best_acc1 = 0
MODELS = ['vit', 'swin', 'cait', 'none']

#############################################
#       Whitening Conv Initialization       #
#############################################

def get_patches(x, patch_shape):
    c, (h, w) = x.shape[1], patch_shape
    return x.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).float()

def get_whitening_parameters(patches):
    n, c, h, w = patches.shape
    patches_flat = patches.view(n, -1)
    est_patch_covariance = (patches_flat.T @ patches_flat) / n
    eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO='U')
    return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.T.reshape(c*h*w, c, h, w).flip(0)

def init_whitening_conv(layer, train_set, eps=5e-4):
    patches = get_patches(train_set, patch_shape=layer.weight.data.shape[2:])
    eigenvalues, eigenvectors = get_whitening_parameters(patches)
    eigenvectors_scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    n_patterns = eigenvectors_scaled.shape[0]
    n_channels = layer.weight.data.shape[0]
    layer.weight.data[:n_patterns] = eigenvectors_scaled

def init_parser():
    parser = argparse.ArgumentParser(
        description='Vit small datasets quick training script')
    parser.add_argument('--profile', action='store_true', default=False, help='profile training')
    parser.add_argument('--whitening', action='store_true', default=False, help='Use whitening initialization')
    parser.add_argument('--sin_pos', action='store_true', default=False, help='Use sin position embedding')
    parser.add_argument('--gin', nargs='+', type=str, default=[], help='Configure Modules')
    # Data args
    parser.add_argument('--datapath', default='./data', type=str, help='dataset path')
    parser.add_argument('--dataset', default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'Tiny-Imagenet', 'SVHN', 'CINIC'], type=str, help='small dataset path')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='number of data loading workers (default: 10)')
    parser.add_argument('--print-freq', default=1, type=int, metavar='N', help='log frequency (by iteration)')
    # Optimization hyperparams
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--warmup', default=10, type=int, metavar='N', help='number of warmup epochs')
    parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)', dest='batch_size')
    parser.add_argument('--opt', default='adamw', type=str, help='optimizer')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--weight-decay', default=5e-2, type=float, help='weight decay (default: 5e-2)')
    parser.add_argument('--arch', type=str, default='vit', choices=MODELS)
    parser.add_argument('--disable-cos', action='store_true', help='disable cosine lr schedule')
    parser.add_argument('--enable_aug', action='store_true', help='disable augmentation policies for training')
    # GPU 관련 인자는 DP 사용을 위해 기본 GPU 0로 설정합니다.
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--amp', action='store_true', help='enable mixed precision')
    parser.add_argument('--no_cuda', action='store_true', help='disable cuda')
    parser.add_argument('--ls', action='store_false', help='label smoothing')
    parser.add_argument('--channel', type=int, help='channel')
    parser.add_argument('--heads', type=int, help='heads')
    parser.add_argument('--depth', type=int, help='depth')
    parser.add_argument('--tag', type=str, help='tag', default='')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--sd', default=0.1, type=float, help='rate of stochastic depth')
    parser.add_argument('--resume', default=False, help='Resume checkpoint path')
    parser.add_argument('--aa', action='store_false', help='Auto augmentation used')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--cm', action='store_false', help='Use Cutmix')
    parser.add_argument('--beta', default=1.0, type=float, help='hyperparameter beta (default: 1)')
    parser.add_argument('--mu', action='store_false', help='Use Mixup')
    parser.add_argument('--alpha', default=1.0, type=float, help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('--mix_prob', default=0.5, type=float, help='mixup probability')
    parser.add_argument('--ra', type=int, default=3, help='repeated augmentation')
    parser.add_argument('--re', default=0.25, type=float, help='Random Erasing probability')
    parser.add_argument('--re_sh', default=0.4, type=float, help='max erasing area')
    parser.add_argument('--re_r1', default=0.3, type=float, help='aspect of erasing area')
    parser.add_argument('--is_LSA', action='store_true', help='Locality Self-Attention')
    parser.add_argument('--is_SPT', action='store_true', help='Shifted Patch Tokenization')
    # 여기서 --pretrained_weights 인자를 통해 train_ssl.py에서 저장된 가중치 파일 경로를 전달할 수 있습니다.
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights from train_ssl.py")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--patch_size', default=4, type=int, help='patch size for ViT')
    parser.add_argument('--vit_mlp_ratio', default=2, type=int, help='MLP layers in the transformer encoder')
    return parser

# 추가: test 함수 (다운스트림 평가)
def test(epoch, testloader, model, criterion, device):
    """
    test 함수는 모델을 평가합니다.
    DP 환경에서 model.eval() 후 testloader로 손실과 분류 정확도를 계산합니다.
    """
    global best_acc1
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader),
                         'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print()
    acc = 100. * correct / total
    if acc > best_acc1:
        best_acc1 = acc
        # 최고 모델 저장: train_ssl.py에서 저장한 가중치를 로드한 경우, fine-tuning 시에도 동일 가중치를 활용합니다.
        torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
    print('Test Epoch: {} | Loss: {:.3f} | Acc: {:.3f}%'.format(epoch, test_loss/len(testloader), acc))
    return best_acc1

def main(args):
    # GPU 설정: DP에서는 단일 GPU에 국한하지 않고 전체 사용
    torch.cuda.set_device(args.gpu)

    global logger
    data_info = datainfo(logger, args)

    model = create_model(data_info['img_size'], data_info['n_classes'], args)
    model.cuda()  # 모델을 GPU로 이동

    # DDP 대신 DataParallel로 wrapping (모든 사용 가능한 GPU 사용)
    model = nn.DataParallel(model)
    model_without_dp = model.module

    print(Fore.GREEN + '*' * 80)
    logger.debug(f"Creating model: {args.arch}")
    print(model_without_dp)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.debug(f'Number of params: {format(n_parameters, ",")}')
    logger.debug(f'Initial learning rate: {args.lr:.6f}')
    logger.debug(f"Start training for {args.epochs} epochs")
    print('*' * 80 + Style.RESET_ALL)

    # pretrained_weights 인자가 전달된 경우, train_ssl.py에서 저장한 가중치를 로드합니다.
    if args.pretrained_weights and os.path.isfile(args.pretrained_weights):
        model_dict = model.state_dict()
        print("Loading pretrained weights from:", args.pretrained_weights)
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Using key {args.checkpoint_key} from the checkpoint")
            state_dict = state_dict[args.checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        state_dict = {k: v if v.size() == model_dict[k].size() else model_dict[k] 
                      for k, v in zip(model_dict.keys(), state_dict.values())}
        model.load_state_dict(state_dict, strict=False)

    if args.ls:
        print(Fore.YELLOW + '*' * 80)
        logger.debug('Label smoothing used')
        print('*' * 80 + Style.RESET_ALL)
        criterion = LabelSmoothingCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    if args.sd > 0.:
        print(Fore.YELLOW + '*' * 80)
        logger.debug(f'Stochastic depth({args.sd}) used')
        print('*' * 80 + Style.RESET_ALL)

    normalize = [transforms.Normalize(mean=data_info['stat'][0], std=data_info['stat'][1])]

    if args.cm:
        print(Fore.YELLOW + '*' * 80)
        logger.debug('Cutmix used')
        print('*' * 80 + Style.RESET_ALL)
    if args.mu:
        print(Fore.YELLOW + '*' * 80)
        logger.debug('Mixup used')
        print('*' * 80 + Style.RESET_ALL)
    if args.ra > 1:
        print(Fore.YELLOW + '*' * 80)
        logger.debug(f'Repeated Aug({args.ra}) used')
        print('*' * 80 + Style.RESET_ALL)

    '''
        Data Augmentation
    '''
    augmentations = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(data_info['img_size'], padding=4)
    ]
    if args.aa:
        print(Fore.YELLOW + '*' * 80)
        logger.debug('Autoaugmentation used')
        if 'CIFAR' in args.dataset:
            print("CIFAR Policy")
            from utils.autoaug import CIFAR10Policy
            augmentations += [CIFAR10Policy()]
        elif 'SVHN' in args.dataset:
            print("SVHN Policy")
            from utils.autoaug import SVHNPolicy
            augmentations += [SVHNPolicy()]
        else:
            from utils.autoaug import ImageNetPolicy
            augmentations += [ImageNetPolicy()]
        print('*' * 80 + Style.RESET_ALL)
    augmentations += [transforms.ToTensor(), *normalize]
    if args.re > 0:
        from utils.random_erasing import RandomErasing
        print(Fore.YELLOW + '*' * 80)
        logger.debug(f'Random erasing({args.re}) used')
        print('*' * 80 + Style.RESET_ALL)
        augmentations += [RandomErasing(probability=args.re, sh=args.re_sh,
                                        r1=args.re_r1, mean=data_info['stat'][0])]
    augmentations = transforms.Compose(augmentations)

    # 데이터 로더: 분산 샘플러 대신 일반 DataLoader 사용
    train_dataset, val_dataset = dataload(args, augmentations, normalize, data_info)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        pin_memory=True, num_workers=args.workers)

    # test 함수에서 사용할 testloader를 val_loader로 사용 (필요 시 별도 분리 가능)
    global testloader
    testloader = val_loader

    if args.whitening:
        with torch.no_grad():
            train_images = []
            for x, y in train_loader:
                train_images.append(x)
                if len(train_images) * len(x) > 5000:
                    break
            train_images = torch.cat(train_images)
            init_whitening_conv(model_without_dp.patch_embed.proj, train_images)

    if args.opt == 'muon':
        from utils.muon import Muon
        muon_params = []
        adamw_params = []
        for name, p in model.named_parameters():
            if 'head' in name or 'fc' in name or 'embed' in name:
                adamw_params.append(p)
            elif p.ndim >= 2:
                muon_params.append(p)
            else:
                adamw_params.append(p)
        print(f"Muon parameters: {len(muon_params)}")
        optimizer = Muon(muon_params, lr=0.02, momentum=0.9,
                         adamw_params=adamw_params, adamw_lr=1e-3, adamw_wd=args.weight_decay)
    elif args.opt == 'lion':
        from lion_pytorch import Lion
        optimizer = Lion(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(args, optimizer, len(train_loader))

    print()
    print("Beginning training")
    print()

    lr_current = optimizer.param_groups[0]["lr"]
    if args.amp:
        loss_scaler = torch.cuda.amp.GradScaler()
    else:
        loss_scaler = None

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        final_epoch = args.epochs
        args.epochs = final_epoch - (checkpoint['epoch'] + 1)

    pbar = tqdm(range(args.epochs))
    
    for epoch in pbar:
        lr_current = train(train_loader, model, criterion, optimizer,
                           epoch, scheduler, loss_scaler, args)
        # validate() 함수 기존 호출 (매 5 에포크마다)
        if epoch % 5 == 0:
            acc1 = validate(val_loader, model, criterion, lr_current, args, epoch=epoch)
            logger_dict.print()
            global best_acc1
            if acc1 > best_acc1:
                print('* Best model update *')
                best_acc1 = acc1
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, os.path.join(save_path, 'best.pth'))
                print(f'Best acc1 {best_acc1:.2f}')
                print('*' * 80)
        # 추가: 매 에포크마다 test 함수 호출
        test_acc = test(epoch, testloader, model, criterion, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        pbar.set_description(f"Epoch {epoch} | Test Acc: {test_acc:.2f}%")
    print(Fore.RED + '*' * 80)
    logger.debug(f'best top-1: {best_acc1:.2f}')
    print('*' * 80 + Style.RESET_ALL)
    torch.save(model.state_dict(), os.path.join(save_path, 'checkpoint.pth'))

def train(train_loader, model, criterion, optimizer, epoch, scheduler, loss_scaler, args):
    model.train()
    loss_val, acc1_val = 0, 0
    n = 0

    for i, (images, target) in enumerate(train_loader):
        if (not args.no_cuda) and torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # Cutmix only
        if args.cm and not args.mu:
            r = np.random.rand(1)
            with torch.cuda.amp.autocast(enabled=args.amp):
                if r < args.mix_prob:
                    slicing_idx, y_a, y_b, lam, sliced = cutmix_data(images, target, args)
                    images[:, :, slicing_idx[0]:slicing_idx[2],
                           slicing_idx[1]:slicing_idx[3]] = sliced
                    output = model(images)
                    loss = mixup_criterion(criterion, output, y_a, y_b, lam)
                else:
                    output = model(images)
                    loss = criterion(output, target)
        # Mixup only
        elif not args.cm and args.mu:
            r = np.random.rand(1)
            if r < args.mix_prob:
                images, y_a, y_b, lam = mixup_data(images, target, args)
                output = model(images)
                loss = mixup_criterion(criterion, output, y_a, y_b, lam)
            else:
                output = model(images)
                loss = criterion(output, target)
        # Both Cutmix and Mixup
        elif args.cm and args.mu:
            r = np.random.rand(1)
            if r < args.mix_prob:
                switching_prob = np.random.rand(1)
                if switching_prob < 0.5:
                    slicing_idx, y_a, y_b, lam, sliced = cutmix_data(images, target, args)
                    images[:, :, slicing_idx[0]:slicing_idx[2],
                           slicing_idx[1]:slicing_idx[3]] = sliced
                    output = model(images)
                    loss = mixup_criterion(criterion, output, y_a, y_b, lam)
                else:
                    images, y_a, y_b, lam = mixup_data(images, target, args)
                    output = model(images)
                    loss = mixup_criterion(criterion, output, y_a, y_b, lam)
            else:
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc = accuracy(output, target, (1,))
        acc1 = acc[0]
        n += images.size(0)
        loss_val += float(loss.item() * images.size(0))
        acc1_val += float(acc1[0] * images.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler.scale(loss).backward()
            loss_scaler.step(optimizer)
            loss_scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()  # update learning rate
        lr_current = optimizer.param_groups[0]["lr"]
        if wandb.run:
            wandb.log({"loss": loss.item(), "lr": lr_current})
    avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
    logger_dict.update(keys[0], avg_loss)
    logger_dict.update(keys[1], avg_acc1)
    if wandb.run:
        wandb.log({"train/loss": avg_loss, "train/acc": avg_acc1, 'epoch': epoch})
    return lr_current

def validate(val_loader, model, criterion, lr, args, epoch=None):
    model.eval()
    loss_val, acc1_val = 0, 0
    n = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if (not args.no_cuda) and torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            acc = accuracy(output, target, (1, 5))
            acc1 = acc[0]
            n += images.size(0)
            loss_val += float(loss.item() * images.size(0))
            acc1_val += float(acc1[0] * images.size(0))
            if args.print_freq >= 0 and i % args.print_freq == 0:
                avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
                progress_bar(i, len(val_loader),
                             f'[Epoch {epoch+1}][V][{i}]   Loss: {avg_loss:.4e}   Top-1: {avg_acc1:6.2f}   LR: {lr:.6f}')
    print()
    print(Fore.BLUE)
    print('*' * 80)
    logger_dict.update(keys[2], avg_loss)
    logger_dict.update(keys[3], avg_acc1)
    wandb.log({"val/loss": avg_loss, "val/acc": avg_acc1, 'epoch': epoch})
    return avg_acc1

def train_profile(train_loader, model, criterion, optimizer, epoch, scheduler, loss_scaler, args):
    model.train()
    loss_val, acc1_val = 0, 0
    n = 0
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for i, (images, target) in enumerate(train_loader):
            with torch.profiler.record_function("load_batch"):
                if (not args.no_cuda) and torch.cuda.is_available():
                    images = images.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
            with torch.profiler.record_function("forward"):
                r = np.random.rand(1)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    if r < args.mix_prob:
                        slicing_idx, y_a, y_b, lam, sliced = cutmix_data(images, target, args)
                        images[:, :, slicing_idx[0]:slicing_idx[2],
                               slicing_idx[1]:slicing_idx[3]] = sliced
                        output = model(images)
                        loss = mixup_criterion(criterion, output, y_a, y_b, lam)
                    else:
                        output = model(images)
                        loss = criterion(output, target)
            with torch.profiler.record_function("backward"):
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            prof.step()
            if i > 10:
                break

if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    gin.parse_config(args.gin)
    # DP 사용을 위해 분산 관련 초기화 제거
    args.gpu = 0
    # 학습 결과를 저장할 경로 설정
    global save_path
    save_path = os.path.join(os.getcwd(), 'save_finetuned', f"{args.arch}-{args.tag}-{args.dataset}-LR[{args.lr}]-Seed{args.seed}")
    os.makedirs(save_path, exist_ok=True)
    wandb.init(job_type='finetune', dir=save_path, config=args.__dict__, tensorboard=True)
    log_dir = os.path.join(save_path, 'history.csv')
    logger = log.getLogger(__name__)
    formatter = log.Formatter('%(message)s')
    streamHandler = log.StreamHandler()
    fileHandler = log.FileHandler(log_dir, 'a')
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(level=log.DEBUG)
    global logger_dict
    global keys
    logger_dict = Logger_dict(logger, save_path)
    keys = ['T Loss', 'T Top-1', 'V Loss', 'V Top-1']
    main(args)
