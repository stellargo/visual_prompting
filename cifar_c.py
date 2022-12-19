from __future__ import print_function

import argparse
import os
from tqdm import tqdm
import time
import random
import wandb
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, ImageFolder
import torchvision.transforms as transforms

import clip
from models import prompters
from utils import accuracy, AverageMeter, ProgressMeter, save_checkpoint
from utils import cosine_lr, convert_models_to_fp32, refine_classname


def parse_option():
    parser = argparse.ArgumentParser('Visual Prompting for CLIP')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epoch5s')

    # optimization
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=40,
                        help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="weight decay")
    parser.add_argument("--warmup", type=int, default=1000,
                        help="number of steps to warmup for")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--patience', type=int, default=1000)
    parser.add_argument('--train_folder', type=str, help="train folder path")
    parser.add_argument('--val_folder', type=str, help="val folder path")

    # model
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--arch', type=str, default='vit_b32')
    parser.add_argument('--method', type=str, default='padding',
                        choices=['padding', 'random_patch', 'fixed_patch'],
                        help='choose visual prompting method')
    parser.add_argument('--prompt_size', type=int, default=30,
                        help='size for visual prompts')

    # dataset
    parser.add_argument('--root', type=str, default='./data',
                        help='dataset')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='dataset')
    parser.add_argument('--image_size', type=int, default=224,
                        help='image size')

    # other
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for initializing training')
    parser.add_argument('--model_dir', type=str, default='./save/models',
                        help='path to save models')
    parser.add_argument('--image_dir', type=str, default='./save/images',
                        help='path to save images')
    parser.add_argument('--filename', type=str, default=None,
                        help='filename to save')
    parser.add_argument('--trial', type=int, default=1,
                        help='number of trials')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to resume from checkpoint')
    parser.add_argument('--evaluate', default=False,
                        action="store_true",
                        help='evaluate model test set')
    parser.add_argument('--gpu', type=int, default=None,
                        help='gpu to use')
    parser.add_argument('--use_wandb', default=False,
                        action="store_true",
                        help='whether to use wandb')
    parser.add_argument('--cifar_c_path', type=str, default='./save/models',
                        help='path to save models')
    parser.add_argument('--model_saved_path', type=str, default='./save/models',
                        help='path to save models')

    args = parser.parse_args()

    args.filename = '{}_{}_{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}_warmup_{}_trial_{}'. \
        format(args.method, args.prompt_size, args.dataset, args.model, args.arch,
               args.optim, args.learning_rate, args.weight_decay, args.batch_size, args.warmup, args.trial)

    return args


best_acc1 = 0
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    global best_acc1, device

    args = parse_option()
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # create model
    model, preprocess = clip.load('ViT-B/32', device, jit=False)
    checkpoint = torch.load(args.model_saved_path)
    convert_models_to_fp32(model)
    model.eval()

    prompter = prompters.__dict__[args.method](args).to(device)
    prompter.load_state_dict(checkpoint['state_dict'])

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            prompter.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # create data
    template = 'This is a photo of a {}'
    print(f'template: {template}')

    cifar_train_dataset = CIFAR100(args.root, transform=preprocess,
                                   download=True, train=True)

    cifar_val_dataset = CIFAR100(args.root, transform=preprocess,
                                 download=True, train=False)

    # train_dataset = ImageFolder(root=args.train_folder,
    #                             transform=preprocess)
    # val_dataset = ImageFolder(root=args.val_folder,
    #                           transform=preprocess)
    #
    # combined_train_dataset = torch.utils.data.ConcatDataset([cifar_train_dataset, train_dataset])
    # combined_train_dataloader = DataLoader(combined_train_dataset, batch_size=args.batch_size, pin_memory=True,
    #                                        num_workers=args.num_workers, shuffle=True)
    #
    # combined_val_dataset = torch.utils.data.ConcatDataset([cifar_val_dataset, val_dataset])
    # combined_val_dataloader = DataLoader(combined_val_dataset, batch_size=args.batch_size, pin_memory=True,
    #                                      num_workers=args.num_workers, shuffle=False)

    # print(args.batch_size)
    # train_loader = DataLoader(train_dataset,
    #                           batch_size=args.batch_size, pin_memory=False,
    #                           num_workers=args.num_workers, shuffle=True)
    #
    # val_loader = DataLoader(val_dataset,
    #                         batch_size=args.batch_size, pin_memory=False,
    #                         num_workers=args.num_workers, shuffle=False)

    class_names = cifar_train_dataset.classes
    class_names = refine_classname(class_names)
    texts = [template.format(label) for label in class_names]

    # define criterion and optimizer
    optimizer = torch.optim.SGD(prompter.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    scaler = GradScaler()
    # total_steps = len(combined_train_dataloader) * args.epochs
    # total_steps = 1000
    # scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)

    cudnn.benchmark = True

    # make dir
    refined_template = template.lower().replace(' ', '_')
    args.filename = f'{args.filename}_template_{refined_template}'

    args.model_folder = os.path.join(args.model_dir, args.filename)
    if not os.path.isdir(args.model_folder):
        os.makedirs(args.model_folder)

    # wandb
    if args.use_wandb:
        wandb.init(project='Visual Prompting')
        wandb.config.update(args)
        wandb.run.name = args.filename
        wandb.watch(prompter, criterion, log='all', log_freq=10)

    # if args.evaluate:
    #     acc1 = validate(combined_val_dataloader, texts, model, prompter, criterion, args)
    #     return

    epochs_since_improvement = 0

    preprocess = transforms.Compose([
        transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
        transforms.CenterCrop(size=(224, 224)),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    validate(texts, model, prompter, criterion, args, preprocess)

    # for epoch in range(args.epochs):

        # train for one epoch
        # train(combined_train_dataloader, texts, model, prompter, optimizer, scheduler, criterion, scaler, epoch, args)

        # # evaluate on validation set
        # validate(combined_val_dataloader, texts, model, prompter, criterion, args)

        # remember best acc@1 and save checkpoint
        # is_best = acc1 > best_acc1
        # best_acc1 = max(acc1, best_acc1)
        #
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': prompter.state_dict(),
        #     'best_acc1': best_acc1,
        #     'optimizer': optimizer.state_dict(),
        # }, args, is_best=is_best)
        #
        # if is_best:
        #     epochs_since_improvement = 0
        # else:
        #     epochs_since_improvement += 1
        #     print(f"There's no improvement for {epochs_since_improvement} epochs.")
        #
        #     if epochs_since_improvement >= args.patience:
        #         print("The training halted by early stopping criterion.")
        #         break

    wandb.run.finish()


# def train(train_loader, texts, model, prompter, optimizer, scheduler, criterion, scaler, epoch, args):
#     batch_time = AverageMeter('Time', ':6.3f')
#     data_time = AverageMeter('Data', ':6.3f')
#     losses = AverageMeter('Loss', ':.4e')
#     top1 = AverageMeter('Acc@1', ':6.2f')
#     progress = ProgressMeter(
#         len(train_loader),
#         [batch_time, data_time, losses, top1],
#         prefix="Epoch: [{}]".format(epoch))
#
#     # switch to train mode
#     prompter.train()
#
#     num_batches_per_epoch = len(train_loader)
#
#     end = time.time()
#     for i, (images, target) in enumerate(tqdm(train_loader)):
#         # print(images.shape)
#
#         # measure data loading time
#         data_time.update(time.time() - end)
#
#         # adjust learning rate
#         step = num_batches_per_epoch * epoch + i
#         scheduler(step)
#
#         optimizer.zero_grad()
#
#         images = images.to(device)
#         target = target.to(device)
#         text_tokens = clip.tokenize(texts).to(device)
#
#         # with automatic mixed precision
#         with autocast():
#             prompted_images = prompter(images)
#
#             output, _ = model(prompted_images, text_tokens)
#             # print("Sleeping")
#             # time.sleep(10)
#             # continue
#             loss = criterion(output, target)
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#         scaler.update()
#
#         # Note: we clamp to 4.6052 = ln(100), as in the original paper.
#         model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)
#
#         # measure accuracy
#         acc1 = accuracy(output, target, topk=(1,))
#         losses.update(loss.item(), images.size(0))
#         top1.update(acc1[0].item(), images.size(0))
#
#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         if i % args.print_freq == 0:
#             progress.display(i)
#
#             if args.use_wandb:
#                 wandb.log({
#                     'training_loss': losses.avg,
#                     'training_acc': top1.avg
#                 })
#
#         if i % args.save_freq == 0:
#             save_checkpoint({
#                 'epoch': epoch + 1,
#                 'state_dict': prompter.state_dict(),
#                 'best_acc1': best_acc1,
#                 'optimizer': optimizer.state_dict(),
#             }, args)
#
#     return losses.avg, top1.avg


def validate(texts, model, prompter, criterion, args, preprocess):

    corruptions = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur",
                   "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
                   "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur"]

    with torch.no_grad():
        targets = np.load(args.cifar_c_path + '/labels.npy')

        for corruption in corruptions:
            data_cifar = np.load(args.cifar_c_path + '/' + corruption + '.npy')

            for i in range(0, 5):
                print("Evaluating corruption and level : ", corruption, i)

                top1_org = AverageMeter('Original Acc@1', ':6.2f')
                top1_prompt = AverageMeter('Prompt Acc@1', ':6.2f')
                top5_org = AverageMeter('Original Acc@5', ':6.2f')
                top5_prompt = AverageMeter('Prompt Acc@5', ':6.2f')

                # switch to evaluation mode
                prompter.eval()

                for j in tqdm(range(i*10000, (i+1)*10000, args.batch_size)):
                    images = preprocess(torch.Tensor(np.transpose(data_cifar[j:j+args.batch_size], (0, 3, 1, 2)))).to(device)
                    target = torch.Tensor(targets[j:j+args.batch_size]).to(device)
                    text_tokens = clip.tokenize(texts).to(device)
                    prompted_images = prompter(images)

                    output_prompt, _ = model(prompted_images, text_tokens)
                    output_org, _ = model(images, text_tokens)

                    # measure accuracy and record loss
                    acc1 = accuracy(output_prompt, target, topk=(1,))
                    top1_prompt.update(acc1[0].item(), images.size(0))

                    acc5 = accuracy(output_prompt, target, topk=(5,))
                    top5_prompt.update(acc5[0].item(), images.size(0))

                    acc1 = accuracy(output_org, target, topk=(1,))
                    top1_org.update(acc1[0].item(), images.size(0))

                    acc5 = accuracy(output_org, target, topk=(5,))
                    top5_org.update(acc5[0].item(), images.size(0))

                print(' * Prompt Acc@1 {top1_prompt.avg:.3f} Original Acc@1 {top1_org.avg:.3f} Prompt Acc@5 {top5_prompt.avg:.3f} Original Acc@5 {top5_org.avg:.3f}'.format(
                    top1_prompt=top1_prompt, top1_org=top1_org, top5_prompt=top5_prompt, top5_org=top5_org))




        # for i, (images, target) in enumerate(tqdm(val_loader)):
        #
        #     images = images.to(device)
        #     target = target.to(device)
        #     text_tokens = clip.tokenize(texts).to(device)
        #     prompted_images = prompter(images)
        #
        #     # compute output
        #     output_prompt, _ = model(prompted_images, text_tokens)
        #     output_org, _ = model(images, text_tokens)
        #     loss = criterion(output_prompt, target)
        #
        #     # measure accuracy and record loss
        #     acc1 = accuracy(output_prompt, target, topk=(1,))
        #     losses.update(loss.item(), images.size(0))
        #     top1_prompt.update(acc1[0].item(), images.size(0))
        #
        #     acc1 = accuracy(output_org, target, topk=(1,))
        #     top1_org.update(acc1[0].item(), images.size(0))
        #
        #     # measure elapsed time
        #     batch_time.update(time.time() - end)
        #     end = time.time()
        #
        #     # if i % args.print_freq == 0:
        #     #     progress.display(i)

        # print(' * Prompt Acc@1 {top1_prompt.avg:.3f} Original Acc@1 {top1_org.avg:.3f}'.format(top1_prompt=top1_prompt, top1_org=top1_org))
        #
        # if args.use_wandb:
        #     wandb.log({
        #         'val_loss': losses.avg,
        #         'val_acc_prompt': top1_prompt.avg,
        #         'val_acc_org': top1_org.avg,
        #     })

    return top1_prompt.avg


if __name__ == '__main__':
    main()