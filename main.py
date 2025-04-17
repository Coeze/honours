import argparse
import os
from src.models2 import resnet20
from src.loss import DynamicRoutingLoss
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from src.datasets import get_test_loader, get_train_valid_loader, DATASET_CONFIGS
from src.models import ETCaps, ResNetCaps
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

# Helper function to get available GPUs
def get_available_gpus():
    """Return the number of available GPUs on this system"""
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        return n_gpus
    else:
        return 0

def train(model, train_loader, val_loader, epochs, scheduler, use_tensorboard, optimizer, device, rank=-1):
    """
    Train the model on the training set.

    A checkpoint of the model is saved after each epoch
    and if the validation accuracy is improved upon,
    a separate ckpt is created for use on the test set.
    """
    # load the most recent checkpoint
    if use_tensorboard and (rank == 0 or rank == -1):
        writer = SummaryWriter(log_dir=os.path.join('./tensorboard', 'logs'))
    if rank == 0 or rank == -1:
        print("\n[*] Train on {} samples, validate on {} samples".format(
            len(train_loader.dataset), len(val_loader.dataset))
        )
    best_valid_acc = 0 
    
    loss_fn = DynamicRoutingLoss()
    counter = 0

    for epoch in range(0, epochs):
        # get current lr
        for i, param_group in enumerate(optimizer.param_groups):
            lr = float(param_group['lr'])
            break

        if rank == 0 or rank == -1:
            print(
                    '\nEpoch: {}/{} - LR: {:.1e}'.format(epoch+1, epochs, lr)
            )

        # train for 1 epoch
        train_loss, train_acc = train_one_epoch(train_loader, device, epochs, loss_fn, model, optimizer, use_tensorboard, rank)

        # evaluate on validation set
        with torch.no_grad():
            valid_loss, valid_acc = validate(model, epoch, loss_fn, device, val_loader, writer if use_tensorboard and (rank == 0 or rank == -1) else None, use_tensorboard, rank)

        if rank == 0 or rank == -1:
            msg1 = "train loss: {:.3f} - train acc: {:.3f}"
            msg2 = " - val loss: {:.3f} - val acc: {:.3f}"

            is_best = valid_acc > best_valid_acc
            if is_best:
                counter = 0
                msg2 += " [*]"

            msg = msg1 + msg2
            print(msg.format(train_loss, train_acc, valid_loss, valid_acc))

            # check for improvement
            if not is_best:
                counter += 1
            '''
            if counter > train_patience:
                print("[!] No improvement in a while, stopping training.")
                return
            '''

            best_valid_acc = max(valid_acc, best_valid_acc)

            # Only save checkpoints on rank 0
            if rank == 0 or rank == -1:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_valid_acc': best_valid_acc
                }

                # Save the checkpoint to a file
                torch.save(checkpoint, 'best.pth')

        # decay lr
        scheduler.step()

    if use_tensorboard and (rank == 0 or rank == -1):
        writer.close()

    if rank == 0 or rank == -1:
        print(best_valid_acc)

def train_one_epoch(train_loader, device, epoch, loss_fn, model, optimizer, use_tensorboard, rank=-1):
    """
    Train the model for 1 epoch of the training set.

    An epoch corresponds to one full pass through the entire
    training set in successive mini-batches.

    This is used by train() and should not be called manually.
    """
    model.train()

    if use_tensorboard and (rank == 0 or rank == -1):
        writer = SummaryWriter(log_dir=os.path.join('./tensorboard', 'logs'))

    # Replace AverageMeter with simple variables
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    tic = time.time()
    
    # Only use tqdm on rank 0 or when not using distributed training
    if rank == 0 or rank == -1:
        pbar = tqdm(total=len(train_loader.dataset))
    
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        b = x.shape[0]
        out = model(x)
        loss = loss_fn(out, y)

        # compute accuracy
        pred = torch.max(out, 1)[1]
        correct = (pred == y).float()
        acc = 100 * (correct.sum() / len(y))

        # Update running statistics
        total_loss += loss.data.item() * b
        total_acc += acc.data.item() * b
        total_samples += b
        
        # Calculate current average
        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples

        # compute gradients and update SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Only update progress bar on rank 0 or when not using distributed training
        if rank == 0 or rank == -1:
            toc = time.time()
            pbar.set_description(
                (
                    "{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
                        (toc-tic), loss.data.item(), acc.data.item()
                    )
                )
            )
            pbar.update(b)

        if use_tensorboard and (rank == 0 or rank == -1):
            iteration = epoch*len(train_loader) + i
            writer.add_scalar('train_loss', loss, iteration)
            writer.add_scalar('train_acc', acc, iteration)

    if rank == 0 or rank == -1:
        pbar.close()

    # If using distributed training, we need to synchronize the statistics across all processes
    if dist.is_initialized():
        # Create a tensor for each stat on the current device
        loss_tensor = torch.tensor([total_loss], device=device)
        acc_tensor = torch.tensor([total_acc], device=device)
        samples_tensor = torch.tensor([total_samples], device=device)
        
        # Sum across all processes
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
        
        # Update variables with synchronized values
        total_loss = loss_tensor.item()
        total_acc = acc_tensor.item()
        total_samples = samples_tensor.item()

    # Calculate final averages
    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples

    return avg_loss, avg_acc

def validate(model, epoch, loss_fn, device, valid_loader, writer, use_tensorboard=False, rank=-1):
    """
    Evaluate the model on the validation set.
    """
    model.eval()

    # Replace AverageMeter with simple variables
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for i, (x, y) in enumerate(valid_loader):
        x, y = x.to(device), y.to(device)

        out = model(x)
        loss = loss_fn(out, y)

        # compute accuracy
        pred = torch.max(out, 1)[1]
        correct = (pred == y).float()
        acc = 100 * (correct.sum() / len(y))

        # Update running statistics
        batch_size = x.size()[0]
        total_loss += loss.data.item() * batch_size
        total_acc += acc.data.item() * batch_size
        total_samples += batch_size

    # If using distributed training, we need to synchronize the statistics across all processes
    if dist.is_initialized():
        # Create a tensor for each stat on the current device
        loss_tensor = torch.tensor([total_loss], device=device)
        acc_tensor = torch.tensor([total_acc], device=device)
        samples_tensor = torch.tensor([total_samples], device=device)
        
        # Sum across all processes
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
        
        # Update variables with synchronized values
        total_loss = loss_tensor.item()
        total_acc = acc_tensor.item()
        total_samples = samples_tensor.item()

    # Calculate final averages
    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples

    # log to tensorboard (only on rank 0 or when not using distributed training)
    if use_tensorboard and writer is not None and (rank == 0 or rank == -1):
        writer.add_scalar('valid_loss', avg_loss, epoch)
        writer.add_scalar('valid_acc', avg_acc, epoch)

    return avg_loss, avg_acc

def test(model_pth, test_loader, device):
    """
    Test the model on the held-out test data.
    This function should only be called at the very
    end once the model has finished training.
    """
    correct = 0

    # load the best checkpoint
    model = torch.load(torch.load_state_dict(model_pth))
    model.eval()

    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)

        out = model(x)

        # compute accuracy
        pred = torch.max(out, 1)[1]
        correct += pred.eq(y.data.view_as(pred)).cpu().sum()

    perc = (100. * correct.data.item()) / len(test_loader.dataset)
    error = 100 - perc
    print(
        '[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)'.format(
            correct, len(test_loader.dataset), perc, error)
    )
    return error

def setup(rank, world_size):
    """
    Setup distributed training
    """
    # set the communication method
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """
    Clean up the distributed environment
    """
    dist.destroy_process_group()

def distributed_train(rank, world_size, args):
    # setup the process group
    setup(rank, world_size)
    
    # set the device
    device = torch.device(f"cuda:{rank}")
    
    # prepare data loader
    train_sampler, train_loader, valid_loader = get_distributed_loaders(
        args.data_dir, args.dataset, args.batch_size, args.random_seed, world_size, rank
    )
    
    # create model and move it to GPU with id rank
    if args.et:
        model = resnet20(planes=512, cfg_data=DATASET_CONFIGS[args.dataset], num_caps=32, caps_size=8, depth=2).to(device)
    else:
        model = ResNetCaps(in_channels=DATASET_CONFIGS[args.dataset]['channels']).to(device)
    
    # wrap the model with DDP
    model = DDP(model, device_ids=[rank])
    
    # create optimizer and scheduler
    params = model.parameters()
    optimizer = optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=5e-4)
    if args.dataset == "cifar10":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)
    elif args.dataset == "svhn":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    elif args.dataset == "smallnorb":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    
    # train the model
    train(
        model,
        train_loader,
        valid_loader,
        args.epochs,
        scheduler,
        False,
        optimizer,
        device,
        rank
    )
    
    cleanup()

def get_distributed_loaders(data_dir, dataset, batch_size, random_seed, world_size, rank):
    """
    Get data loaders for distributed training
    """
    from torch.utils.data.distributed import DistributedSampler
    
    train_dataset, valid_dataset = get_train_valid_dataset(data_dir, dataset, random_seed)
    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_sampler, train_loader, valid_loader

def get_train_valid_dataset(data_dir, dataset, random_seed):
    """
    Helper to get datasets instead of loaders for distributed training
    """
    from src.datasets import get_train_valid_dataset
    return get_train_valid_dataset(data_dir, dataset, random_seed)

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
    parser.add_argument(
        "--distributed", action="store_true", help="Use distributed training if available"
    )
    parser.add_argument(
        "--world_size", type=int, default=None, 
        help="Number of GPUs to use for distributed training. If not specified, all available GPUs will be used."
    )

    args = parser.parse_args()

    if args.test:
        print("testing the model...")
        errors = []
        test_loader = get_test_loader(args.data_dir, args.dataset, args.batch_size)
        
        # For testing, we don't need distributed processing
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.et:
            model = resnet20(planes=512, cfg_data=DATASET_CONFIGS[args.dataset], num_caps=32, caps_size=8, depth=2).to(device)
        else:
            model = ResNetCaps(in_channels=DATASET_CONFIGS[args.dataset]['channels']).to(device)
            
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
        
        # Check if distributed training is possible and requested
        n_gpus = get_available_gpus()
        
        # Determine world_size based on available GPUs and args
        if args.distributed and n_gpus > 0:
            # If world_size is not specified, use all available GPUs
            world_size = args.world_size if args.world_size is not None else n_gpus
            
            # Ensure we don't try to use more GPUs than are available
            world_size = min(world_size, n_gpus)
            
            print(f"Using distributed training with {world_size} GPUs")
            
            # Use distributed training with detected GPUs
            mp.spawn(
                distributed_train,
                args=(world_size, args),
                nprocs=world_size,
                join=True
            )
        else:
            if args.distributed and n_gpus == 0:
                print("Distributed training requested but no GPUs available. Falling back to CPU/single GPU.")
            
            # Use single GPU or CPU training
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_loader, valid_loader = get_train_valid_loader(
                args.data_dir, args.dataset, args.batch_size, args.random_seed
            )
            
            if args.et:
                model = resnet20(planes=64, cfg_data=DATASET_CONFIGS[args.dataset], num_caps=32, caps_size=8, depth=2).to(device)
            else:
                model = ResNetCaps(in_channels=DATASET_CONFIGS[args.dataset]['channels']).to(device)
            
            params = model.parameters()
            optimizer = optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=5e-4)
            if args.dataset == "cifar10":
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)
            elif args.dataset == "svhn":
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
            elif args.dataset == "smallnorb":
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
            
            train(
                model,
                train_loader,
                valid_loader,
                args.epochs,
                scheduler,
                False,
                optimizer,
                device
            )


if __name__ == "__main__":
    main()
