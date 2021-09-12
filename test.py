# Based on https://github.com/pytorch/examples/tree/master/imagenet
import os
import sys
import random
import shutil
import logging
import argparse

from tqdm import tqdm, trange

import numpy as np

from scipy.stats import pearsonr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

from models import classifier
from utils.dataset import MAHNOBHCIDataset, VIPLHRDataset, UBFCDataset
from utils.utils import AverageMeter
from utils.augmentation import Transformer, RandomROI

parser = argparse.ArgumentParser()

# Training setting
parser.add_argument("--gpu", default=None, type=int)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument(
    "--epochs", default=50, type=int, help="number of total epochs to run"
)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
parser.add_argument("--wd", default=0, type=float, help="weight decay")
parser.add_argument("--dropout", default=0.0, type=float, help="ResNet dropout value")

# Test setting
parser.add_argument("--evaluate", action="store_true")
parser.add_argument("--scratch", action="store_true")
parser.add_argument(
    "--pretrained", default="", type=str, help="path to pretrained checkpoint"
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
parser.add_argument("--vid_frame", default=150, type=int)
parser.add_argument("--vid_frame_stride", default=2, type=int)

# Log setting
parser.add_argument("--log_dir", default="./path/to/your/log", type=str)
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--run_tag", nargs="+", default=None)
parser.add_argument("--run_name", default=None, type=str)

# Model setting
parser.add_argument("--model_depth", default=18, type=int)
parser.add_argument("--finetune", default="fc", type=str)


def main():
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logging.basicConfig(
        filename=os.path.join(args.log_dir, "test_output.log"),
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

    # Log config
    if args.wandb:
        import wandb

        wandb.init(
            project="temp",
            notes="Test model",
            tags=args.run_tag,
            name=args.run_name,
            job_type="test",
            dir=args.log_dir,
            config=args,
        )
        args = wandb.config

    # Simply call main_worker function
    try:
        main_worker(args)
    except Exception as e:
        logging.critical(e, exc_info=True)
        print(e)


def main_worker(args):
    best_loss = sys.maxsize

    print("Use GPU: {} for training".format(args.gpu))
    logging.info("Use GPU: {} for training".format(args.gpu))
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)

    # Create SLF-RPM model
    print(
        "\n=> Creating SLF-RPM Classifier Model: 3D ResNet-{}".format(
            args.model_depth
        )
    )
    logging.info(
        "=> Creating SLF-RPM Pretrain Model: 3D ResNet-{}".format(
            args.model_depth
        )
    )
    model = classifier.LinearClsResNet3D(
        model_depth=args.model_depth, n_class=1, dropout=args.dropout
    )

    if args.finetune == "fc":
        # Freeze all layers but the last fc
        for name, param in model.named_parameters():
            if name not in ["encoder_q.fc.weight", "encoder_q.fc.bias"]:
                param.requires_grad = False
        logging.info("=> Finetune only fc layer")

    elif args.finetune == "all":
        logging.info("=> Finetune all model layers")

    else:
        print('Invalid argument "finetune"!')
        logging.critical('Invalid argument "finetune"!')
        sys.exit()

    # Check grad
    print("\n===========Check Grad============")
    for name, param in model.named_parameters():
    	print(name, param.requires_grad)
    print("=================================\n")

    # Load from pretrained model
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> Loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            state_dict = checkpoint["state_dict"]
            for k in list(state_dict.keys()):
                # Retain only encoder and contexter weights
                if k.startswith("module.encoder_q") and not k.startswith(
                    "module.encoder_q.fc"
                ):
                    state_dict[k[len("module.") :]] = state_dict[k]

                elif k.startswith("encoder_q") and not k.startswith("encoder_q.fc"):
                    continue

                elif not k.startswith("fc"):
                    state_dict["encoder_q.{}".format(k)] = state_dict[k]

                # Delete renamed or unused k
                del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {
                "encoder_q.fc.weight",
                "encoder_q.fc.bias",
            }, "Missing keys: {};\n Have: {}".format(
                set(msg.missing_keys), list(state_dict.keys())
            )
            print("=> Loaded pre-trained model '{}'".format(args.pretrained))
            logging.info("=> Loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> Error: No checkpoint found at '{}'".format(args.pretrained))
            logging.critical("=> No checkpoint found at '{}'".format(args.pretrained))
            sys.exit()

    elif args.scratch:
        print("=> Pretrained model does not specify, train from the scratch!")
        logging.info("=> Pretrained model does not specify, train from the scratch!")

    else:
        print(
            "=> Error: Pretrained model does not specify, and scratch does not specify!"
        )
        logging.critical(
            "=> Pretrained model does not specify, and scratch does not specify!"
        )
        sys.exit()

    model = model.to(device)
    print(model)

    # Loss function
    criterion = nn.L1Loss().to(device)

    # Optimise only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    if args.finetune == "fc":
        assert len(parameters) == 2  # fc.weight, fc.bias
    optimiser = optim.Adam(parameters, lr=args.lr, weight_decay=args.wd)

    # Load data
    augmentation = [RandomROI([0])]

    if args.dataset_name == "mahnob-hci":
        augmentation = Transformer(
            augmentation, mean=[0.2796, 0.2394, 0.1901], std=[0.1655, 0.1429, 0.1145]
        )
        train_dataset = MAHNOBHCIDataset(
            args.dataset_dir,
            True,
            augmentation,
            args.vid_frame,
            args.vid_frame_stride,
        )
        val_dataset = MAHNOBHCIDataset(
            args.dataset_dir,
            False,
            augmentation,
            args.vid_frame,
            args.vid_frame_stride,
        )
        assert not [
            i for i in val_dataset.files if i in train_dataset.files
        ], "Train/Val datasets are intersected!"

    elif args.dataset_name == "vipl-hr-v2":
        augmentation = Transformer(
            augmentation, mean=[0.3888, 0.2767, 0.2460], std=[0.2899, 0.2378, 0.2232]
        )
        train_dataset = VIPLHRDataset(
            args.dataset_dir,
            True,
            augmentation,
            args.vid_frame,
            args.vid_frame_stride,
        )
        val_dataset = VIPLHRDataset(
            args.dataset_dir,
            False,
            augmentation,
            args.vid_frame,
            args.vid_frame_stride,
        )

    elif args.dataset_name == "ubfc-rppg":
        augmentation = Transformer(
            augmentation, mean=[0.4642, 0.3766, 0.3744], std=[0.2947, 0.2393, 0.2395]
        )
        train_dataset = UBFCDataset(
            args.dataset_dir,
            True,
            augmentation,
            args.vid_frame,
            args.vid_frame_stride,
        )
        val_dataset = UBFCDataset(
            args.dataset_dir,
            False,
            augmentation,
            args.vid_frame,
            args.vid_frame_stride,
        )

    else:
        print("Unsupported datasets!")
        return

    train_sampler = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    if args.evaluate:
        print("=> Loading checkpoint '{}'".format(args.pretrained))
        logging.info("=> Loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict, strict=True)

        print("=> Loaded pre-trained model '{}'".format(args.pretrained))
        logging.info("=> Loaded pre-trained model '{}'".format(args.pretrained))
        mae, std, rmse, r = validate(val_loader, model, criterion, device)
        print(
            "Evaluation Result\n MAE: {:.4f}; SD: {:.4f}; RMSE: {:.4f}; R: {:.4f};".format(
                mae, std, rmse, r
            )
        )
        logging.info(
            "Evaluation Result\n MAE: {:.4f}; SD: {:.4f}; RMSE: {:.4f}; R: {:.4f};".format(
                mae, std, rmse, r
            )
        )
        return

    # Train model
    for epoch in trange(args.epochs, desc="Epoch"):
        train_loss = train(train_loader, model, criterion, optimiser, device)

        # Evaluate on validation set
        val_loss, std, rmse, r = validate(val_loader, model, criterion, device)
        if args.wandb:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "std": std,
                    "rmse": rmse,
                    "r": r,
                }
            )

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        if is_best:
            state = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "best_loss": best_loss,
            }
            path = os.path.join(args.log_dir, "best_test_model.pth.tar")
            torch.save(state, path)

            print("\nModel saved at epoch {}".format(epoch + 1))
            logging.info("Model saved at epoch {}".format(epoch + 1))

        # Logs
        if args.wandb:
            wandb.run.summary["val_loss"] = best_loss

        print(
            """Test Train Loss: {:.4f}, Test Val Loss/Best: {:.4f}/{:.4f}, 
			Test SD: {:.4f}, Test RMSE: {:.4f}, Test R: {:.4f}""".format(
                train_loss, val_loss, best_loss, std, rmse, r
            )
        )
        logging.info(
            "({}/{}) Test Train Loss: {:.4f}, Test Val Loss/Best: {:.4f}/{:.4f}, Test SD: {:.4f}, Test RMSE: {:.4f}, Test R: {:.4f}".format(
                epoch + 1, args.epochs, train_loss, val_loss, best_loss, std, rmse, r
            )
        )

    if args.wandb:
        shutil.copyfile(
            os.path.join(args.log_dir, "test_output.log"),
            os.path.join(wandb.run.dir, "test_output.log"),
        )


def train(train_loader, model, criterion, optimizer, device):
    losses = AverageMeter("Loss", ":.4e")

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    for videos, targets in tqdm(train_loader, desc="Train Iteration"):
        # Process input
        videos = videos.to(device, non_blocking=True)
        targets = targets.reshape(-1, 1).to(device, non_blocking=True)

        # Compute output
        preds = model(videos)

        # Loss
        loss = criterion(preds, targets)
        losses.update(loss.item(), videos.size(0))

        # Compute gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg


@torch.no_grad()
def validate(val_loader, model, criterion, device):
    maes = AverageMeter("MAE", ":.4e")
    mses = AverageMeter("MSE", ":.4e")
    all_pred = []
    all_true = []
    mse_loss_func = nn.MSELoss()

    # Switch to eval mode
    model.eval()

    for videos, targets in tqdm(val_loader, desc="Val Iteration"):
        # Process input
        videos = videos.to(device, non_blocking=True)
        targets = targets.reshape(-1, 1).to(device, non_blocking=True)

        # Compute output
        preds = model(videos)

        # Loss
        mae = criterion(preds, targets)
        mse = mse_loss_func(preds, targets)

        maes.update(mae.item(), targets.size(0))
        mses.update(mse.item(), targets.size(0))

        all_pred.append(preds.detach().cpu())
        all_true.append(targets.detach().cpu())

    all_pred = torch.cat(all_pred).flatten()
    all_true = torch.cat(all_true).flatten()

    # Mean and Std
    diff = all_pred - all_true
    mean = torch.mean(diff)
    std = torch.std(diff)

    # MSE
    mse_loss = mses.avg

    # RMSE
    rmse_loss = np.sqrt(mse_loss)

    r, _ = pearsonr(all_true, all_pred)

    return maes.avg, std, rmse_loss, r


if __name__ == "__main__":
    main()

