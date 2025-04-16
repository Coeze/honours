import argparse
import os
from src.models2 import resnet20
from src.loss import DynamicRoutingLoss
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.datasets import get_test_loader, get_train_valid_loader, DATASET_CONFIGS
from src.models import ETCaps, ResNetCaps
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

def train_and_validate(
    model,
    train_loader,
    valid_loader,
    loss_fn,
    optimizer,
    scheduler,
    device,
    num_epochs,
    checkpoint_dir,
    use_tensorboard=False,
    resume=False,
    start_epoch=0,
    best_valid_acc=0.0,
    writer=None,
    dataset=None,
):
    """
    Trains and validates the model.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for the training dataset.
        valid_loader: DataLoader for the validation dataset.
        loss_fn: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device to run the training on ('cuda' or 'cpu').
        num_epochs: Total number of epochs to train.
        checkpoint_dir: Directory to save checkpoints.
        use_tensorboard: Whether to use TensorBoard for logging.
        resume: Whether to resume training from a checkpoint.
        start_epoch: Starting epoch number (used when resuming).
        best_valid_acc: Best validation accuracy so far.
        writer: TensorBoard SummaryWriter instance (if use_tensorboard is True).
    """
    if use_tensorboard and writer is None:
        writer = SummaryWriter()
    os.makedirs(checkpoint_dir, exist_ok=True)

    if resume:
        checkpoint_path = os.path.join(checkpoint_dir, 'latest.pth')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optim_state'])
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            start_epoch = checkpoint['epoch']
            best_valid_acc = checkpoint['best_valid_acc']
            print(f"Resumed from checkpoint: epoch {start_epoch}, best_valid_acc {best_valid_acc:.2f}%")
        else:
            print("No checkpoint found. Starting from scratch.")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({'loss': loss.item(), 'acc': 100. * train_correct / total_train})

        avg_train_loss = train_loss / total_train
        train_acc = 100. * train_correct / total_train

        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        total_valid = 0

        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

                valid_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_valid += targets.size(0)
                valid_correct += predicted.eq(targets).sum().item()

        avg_valid_loss = valid_loss / total_valid
        valid_acc = 100. * valid_correct / total_valid

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Valid Loss: {avg_valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%")

        if use_tensorboard:
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Loss/Valid', avg_valid_loss, epoch)
            writer.add_scalar('Accuracy/Valid', valid_acc, epoch)

        is_best = valid_acc > best_valid_acc
        if is_best:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'{dataset}_best.pth'))

        # Save latest checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_valid_acc': best_valid_acc
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'{dataset}_latest.pth'))

        scheduler.step()

    if use_tensorboard:
        writer.close()

    print(f"Training complete. Best validation accuracy: {best_valid_acc:.2f}%")



def test(model, device, test_loader, num_test, checkpoint_path=None):
    correct = 0

    # Load the best checkpoint if provided
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"[*] Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.max(outputs, 1)[1]
            correct += preds.eq(labels.view_as(preds)).cpu().sum()

    perc = (100.0 * correct.item()) / num_test
    error = 100.0 - perc
    print(
        "[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)".format(
            correct.item(), num_test, perc, error
        )
    )
    print(error)
    return error 



def main():
    parser = argparse.ArgumentParser(description="Train ETCaps Model")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to dataset directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cifar10", "svhn", "smallnorb"],
        help="Dataset name",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-2, help="Learning rate"
    )
    parser.add_argument(
        "--save_dir", type=str, default="models/", help="Directory to save models"
    )
    parser.add_argument("--et", action="store_true", help="Use equivariant transformer")
    parser.add_argument("--test", action="store_true", help="Train the model")

    args = parser.parse_args()

    channels = DATASET_CONFIGS[args.dataset]["channels"]
    print(f"Using {channels} channels for {args.dataset} dataset.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader = get_train_valid_loader(
        args.data_dir, args.dataset, args.batch_size, args.random_seed
    )
    print(args.et)
    if args.et:
        print(DATASET_CONFIGS[args.dataset]['channels'])
        # model = resnet20(planes=512, cfg_data=DATASET_CONFIGS[args.dataset], num_caps=32, caps_size=8, depth=2).to(device)
        model = ETCaps(
            in_channels=channels
        ).to(device)
    else:
        model = ResNetCaps(in_channels=channels).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    loss_fn = DynamicRoutingLoss()
    writer = SummaryWriter()
    if args.test:
        print("testing the model...")
        errors = []
        test_loader = get_test_loader(args.data_dir, args.dataset, args.batch_size)
        for seed in [0, 1, 2, 3, 4]:
            args.random_seed = seed
            err = test(
                model,
                device,
                test_loader,
                len(test_loader.dataset),
                os.path.join(args.save_dir, f"etcaps_{args.dataset}_best.pth"),
            )
            errors.append(err)

        mean_error = np.mean(errors)
        std_error = np.std(errors)
        print(f"{mean_error:.2f} ± {std_error} {args.dataset}")
    else:
        print("Training the model...")
        best_valid_accuracy = 0.0
        train_and_validate(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    num_epochs=args.epochs,
    checkpoint_dir=args.save_dir,
    use_tensorboard=True,
    resume=False,
    dataset = args.dataset
)

    writer.close()


if __name__ == "__main__":
    main()
