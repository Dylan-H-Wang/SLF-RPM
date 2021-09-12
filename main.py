# Based on https://github.com/pytorch/examples/tree/master/imagenet
import os
import sys
import shutil
import random
import logging
import argparse

from tqdm import tqdm, trange

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

from models.slf_rpm import SLF_RPM
from utils.dataset import MAHNOBHCIDataset, VIPLHRDataset, UBFCDataset
from utils.utils import accuracy, AverageMeter
from utils.augmentation import Transformer, RandomROI, RandomStride

parser = argparse.ArgumentParser()

# Training setting
parser.add_argument("--gpu", default=None, type=int)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument(
    "--epochs", default=200, type=int, help="number of total epochs to run"
)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
parser.add_argument("--wd", default=0, type=float, help="weight decay")
parser.add_argument("--n_dim", default=2048, type=int, help="Feature dimension")
parser.add_argument(
    "--temperature", default=0.5, type=float, help="Softmax temperature"
)

# Data setting
parser.add_argument("--dataset_name", default="mahnob-hci", type=str)
parser.add_argument("--dataset_dir", default=None, type=str)
parser.add_argument(
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 1)",
)
parser.add_argument("--vid_frame", default=150, type=int, help="number of frames for each raw video")
parser.add_argument("--clip_frame", default=30, type=int, help="number of frames for each video clip after temporal augmentation")
parser.add_argument(
    "--roi_list", nargs="+", default=["0", "1", "2", "3", "4", "5", "6"]
)
parser.add_argument("--stride_list", nargs="+", default=["1", "2", "3", "4", "5"])

# Log setting
parser.add_argument("--log_dir", default="./logs", type=str)
parser.add_argument("--wandb", action="store_true", help="use wandb as log tool.")
parser.add_argument("--run_tag", nargs="+", default=None)
parser.add_argument("--run_name", default=None, type=str)

# Model setting
parser.add_argument("--model_depth", default=18, type=int)


def main():
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logging.basicConfig(
        filename=os.path.join(args.log_dir, "train_output.log"),
        format="[%(asctime)s] %(levelname)s: %(message)s",
        level=logging.DEBUG,
    )

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        logging.info(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )
    else:
        cudnn.benchmark = True

    if args.gpu is None:
        logging.info("You have not specify a GPU, use the default value 0")
        args.gpu = 0

    args.roi_list = [int(i) for i in args.roi_list]
    args.stride_list = [int(i) for i in args.stride_list]

    # Log config
    if args.wandb:
        import wandb

        wandb.init(
            project="SLF-RPM",
            notes="Train the model",
            tags=args.run_tag,
            name=args.run_name,
            job_type="train",
            dir=args.log_dir,
            config=args,
        )
        args = wandb.config

    try:
        main_worker(args)
    except Exception as e:
        logging.critical(e, exc_info=True)
        print(e)


def main_worker(args):
    print("Use GPU: {} for training".format(args.gpu))
    logging.info("Use GPU: {} for training".format(args.gpu))
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)

    # Create SLF-RPM model
    print(
        "\n=> Creating SLF-RPM Pretrain Model: 3D ResNet-{} with MLP".format(
            args.model_depth
        )
    )
    logging.info(
        "=> Creating SLF-RPM Pretrain Model: 3D ResNet-{} with MLP".format(
            args.model_depth
        )
    )
    model = SLF_RPM(
        args.model_depth,
        args.n_dim,
        args.temperature,
        len(args.roi_list),
        len(args.stride_list),
    )
    model = model.to(device)
    print(model)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Optimiser function
    optimiser = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Load data
    augmentation = [RandomROI(args.roi_list)]

    if args.dataset_name == "mahnob-hci":
        augmentation = RandomStride(
            args.stride_list,
            args.clip_frame,
            Transformer(
                augmentation,
                mean=[0.2796, 0.2394, 0.1901],
                std=[0.1655, 0.1429, 0.1145],
            ),
        )
        train_dataset = MAHNOBHCIDataset(
            args.dataset_dir, True, augmentation, args.vid_frame
        )

    elif args.dataset_name == "vipl-hr-v2":
        augmentation = RandomStride(
            args.stride_list,
            args.clip_frame,
            Transformer(
                augmentation,
                mean=[0.3888, 0.2767, 0.2460],
                std=[0.2899, 0.2378, 0.2232],
            ),
        )
        train_dataset = VIPLHRDataset(
            args.dataset_dir, True, augmentation, args.vid_frame
        )

    elif args.dataset_name == "ubfc-rppg":
        augmentation = RandomStride(
            args.stride_list,
            args.clip_frame,
            Transformer(
                augmentation,
                mean=[0.4642, 0.3766, 0.3744],
                std=[0.2947, 0.2393, 0.2395],
            ),
        )
        train_dataset = UBFCDataset(
            args.dataset_dir, True, augmentation, args.vid_frame
        )

    else:
        print("Unsupported datasets!")
        return

    best_loss = sys.maxsize
    best_top1 = 0
    best_top5 = 0

    train_sampler = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    # Train model
    for epoch in trange(args.epochs, desc="Epoch"):
        loss, top1, top5 = train(train_loader, model, criterion, optimiser, device)

        if args.wandb:
            wandb.log(
                {"train_loss": loss, "train_top1_acc": top1, "train_top5_acc": top5}
            )

        is_best = loss <= best_loss
        best_loss = min(loss, best_loss)
        best_top1 = max(top1, best_top1)
        best_top5 = max(top5, best_top5)

        if is_best:
            state = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimiser": optimiser.state_dict(),
            }
            path = os.path.join(args.log_dir, "best_train_model.pth.tar")
            torch.save(state, path)

            logging.info("Model saved at epoch {}".format(epoch + 1))
            print("\nModel saved at epoch {}".format(epoch + 1))

        if (epoch + 1) % 50 == 0:
            best_model = os.path.join(args.log_dir, "best_train_model.pth.tar")
            checkpoint = os.path.join(
                args.log_dir, "best_train_model_before_{}.pth.tar".format((epoch + 1))
            )
            shutil.copyfile(best_model, checkpoint)

            logging.info("Best model before epoch {} is saved".format(epoch + 1))
            print("\nBest model before epoch {} is saved".format(epoch + 1))

        # Logs
        if args.wandb:
            wandb.run.summary["train_loss"] = best_loss
            wandb.run.summary["train_top1_acc"] = best_top1
            wandb.run.summary["train_top5_acc"] = best_top5

        print(
            "Train Loss/Best: {:.4f}/{:.4f}, Train Acc-Top1/Best: {:.4f}/{:.4f}, Train Acc-Top5/Best: {:.4f}/{:.4f}".format(
                loss, best_loss, top1, best_top1, top5, best_top5
            )
        )
        logging.info(
            "({}/{}) Train Loss/Best: {:.4f}/{:.4f}, Train Acc-Top1/Best: {:.4f}/{:.4f}, Train Acc-Top5/Best: {:.4f}/{:.4f}".format(
                epoch + 1,
                args.epochs,
                loss,
                best_loss,
                top1,
                best_top1,
                top5,
                best_top5,
            )
        )

    if args.wandb:
        shutil.copyfile(
            os.path.join(args.log_dir, "train_output.log"),
            os.path.join(wandb.run.dir, "train_output.log"),
        )


def train(train_loader, model, criterion, optimizer, device):
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    model.train()

    for videos, _, label_spatial, label_temporal in tqdm(
        train_loader, desc="Iteration"
    ):
        # Process input
        videos[0] = videos[0].to(device, non_blocking=True)
        videos[1] = videos[1].to(device, non_blocking=True)

        label_spatial = torch.cat(label_spatial, axis=0).to(device, non_blocking=True)
        label_temporal = torch.cat(label_temporal, axis=0).to(device, non_blocking=True)

        # Compute output
        logits, labels, pred_spatial, pred_temporal = model(videos)

        # Contrastive loss
        loss_contrast = criterion(logits, labels)
        loss_spatial = criterion(pred_spatial, label_spatial)
        loss_temporal = criterion(pred_temporal, label_temporal)
        loss = loss_contrast + loss_spatial + loss_temporal

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # Measure accuracy and record loss
        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        losses.update(loss, labels.size(0) * 2)
        top1.update(acc1[0], labels.size(0) * 2)
        top5.update(acc5[0], labels.size(0) * 2)

        # Compute gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg, top1.avg, top5.avg


if __name__ == "__main__":
    main()
