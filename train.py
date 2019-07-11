import datetime
import os
import time
import sys
import subprocess

import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms

from coco_utils import get_coco  # get_coco_kp

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from engine import train_one_epoch, evaluate

import utils
import transforms as T


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")
    dataset, num_classes, label_names = get_coco(
        args.data_path, image_set='train', transforms=get_transform(train=True)
    )
    print(f"Categorizing into {num_classes} classes")
    if args.overfit:
        dataset_test, _, _ = get_coco(
            args.data_path, image_set='train', transforms=get_transform(train=False),
        )
        print("Overfitting to train dataset! Only for debugging")
        assert len(dataset) == len(dataset_test)
    else:
        dataset_test, _, _ = get_coco(
            args.data_path, image_set='val', transforms=get_transform(train=False)
        )

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    # Create summary writer for Tensorboard
    if args.run_name:
        log_dir_path = f"runs/{args.run_name}" if args.run_name else None
        if os.path.isdir(log_dir_path):
            print(f"\nError, summary folder 'runs/{args.run_name}' already exists! Chose another name")
            exit()
    else:
        log_dir_path = None
    writer = SummaryWriter(log_dir=log_dir_path)

    # Add some useful text summaries (Tensorboard uses markdown to render text).
    writer.add_text('Command executed', f"python {' '.join(sys.argv)}")
    writer.add_text('Arguments', str(args).replace(", ", ",  \n").replace("Namespace(", "").replace(")", ""))

    # Add repo status data to summary
    try:
        writer.add_text(
            'Git status',
            subprocess.check_output(
                "git log --name-status HEAD^..HEAD".split()
            ).decode('utf-8').replace('\n', '  \n')
        )
        writer.add_text(
            'Git diff',
            subprocess.check_output(
                "git diff".split()
            ).decode('utf-8').replace('\n', '  \n')
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("\nGit not installed or not running from a repo, summary won't have git data!\n")

    print("Creating model")
    model = torchvision.models.detection.__dict__[args.model](num_classes=num_classes,
                                                              pretrained=args.pretrained)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
    # )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_steps, gamma=args.lr_gamma
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    if args.test_only:
        evaluate(
            model, data_loader_test, 0, writer, args.draw_threshold,
            label_names, args.num_draw_predictions, device=device
        )
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        start_epoch = time.time()
        train_one_epoch(
            model, optimizer, data_loader, device, epoch, args.print_freq, writer, label_names
        )
        print(f"Epoch time {time.time() - start_epoch}")
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step=epoch)
        lr_scheduler.step()

        if args.save_every_num_epochs and epoch % args.save_every_num_epochs == 0:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'label_names': label_names},
                os.path.join(writer.log_dir, 'model_{}.pth'.format(epoch))
            )

        if epoch % args.evaluate_every_num_epochs == 0:
            evaluate(
                model, data_loader_test, epoch, writer, args.draw_threshold,
                label_names, args.num_draw_predictions, device=device
            )

    # Save after training is done
    utils.save_on_master({
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'args': args,
        'label_names': label_names},
        os.path.join(writer.log_dir, 'model_{}_finished.pth'.format(epoch))
    )

    writer.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='LINC Detector Training')

    parser.add_argument('--data-path', default='/mnt/hdd1/lalo/coco_easy', help='dataset')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int)
    parser.add_argument('--epochs', default=13, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')  # noqa
    parser.add_argument('--lr-steps', default=[10, 11], nargs='+', type=int, help='decrease lr every step-size epochs')  # noqa
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')  # noqa
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--aspect-ratio-group-factor', default=0, type=int)
    parser.add_argument('--num-draw-predictions', default=5, type=int, help="How many predictions to draw")
    parser.add_argument(
        '-n',
        '--run-name',
        default=None,
        help='Name this run in order to be able to find it in Tensorboard easily')
    parser.add_argument(
        '--evaluate-every-num-epochs',
        default=2,
        type=int,
        help="How many training epochs to run between each evaluation on the validation set"
    )
    parser.add_argument(
        '--save-every-num-epochs',
        default=None,
        type=int,
        help="How many training epochs to run between each saving of the model to disk"
    )
    parser.add_argument(
        '--draw-threshold', default=0.5,
        type=float,
        help="Draw predicted objects in summary if above this confidence threshold"
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument(
        "--overfit",
        dest="overfit",
        help="Eval and train on the val set, for testing overfitting",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')  # noqa

    args = parser.parse_args()

    main(args)
