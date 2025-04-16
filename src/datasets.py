import numpy as np
from pathlib import Path
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Subset
from .norb import smallNORBViewPoint, smallNORB
from src.diebench import Dataset3DIEBench

DATASET_CONFIGS = {
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'svhn': {'size': 32, 'channels': 3, 'classes': 10},
    'smallnorb': {'size': 32, 'channels': 1, 'classes': 5},
}

VIEWPOINT_EXPS = ['azimuth', 'elevation']

def get_train_valid_loader(data_dir,
                           dataset,
                           batch_size,
                           random_seed,
                           exp='azimuth',
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=0,
                           pin_memory=False):

    data_dir = data_dir + '/' + dataset

    if dataset == "cifar10":
        trans = [transforms.RandomCrop(32, padding=4),
                 transforms.RandomHorizontalFlip(0.5),
                 transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                transform=transforms.Compose(trans))

    elif dataset == "svhn":
        normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                     std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
        trans = [transforms.RandomCrop(32, padding=4),
                 transforms.ToTensor(),
                 normalize]
        dataset = datasets.SVHN(data_dir, split='train', download=True,
                transform=transforms.Compose(trans))
        
    elif dataset == "3diebench":
        dataset_root = Path("3DIEBench")  # Replace with your dataset path
        img_file = dataset_root / "train_images.npy"
        labels_file = dataset_root / "train_labels.npy"
        valid_img_file = dataset_root / "valid_images.npy"
        valid_labels_file = dataset_root / "valid_labels.npy"

        train_dataset = Dataset3DIEBench(
            dataset_root=dataset_root,
            img_file=img_file,
            labels_file=labels_file,
            experience="quat",
            transform=None,
            preload=False,
            num_images = 1
        )

        valid_dataset = Dataset3DIEBench(
            dataset_root=dataset_root,
            valid_img_file=valid_img_file,
            valid_labels_file=valid_labels_file,
            experience="quat",
            transform=None,
            preload=False,
            num_images = 1
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_workers)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=num_workers)

        return train_loader, valid_loader

    elif dataset == "smallnorb":
        trans = [transforms.Resize(48),
                transforms.RandomCrop(32),
                transforms.ColorJitter(brightness=32./255, contrast=0.3),
                transforms.ToTensor(),
                #transforms.Normalize((0.7199,), (0.117,))
                ]
        if exp in VIEWPOINT_EXPS:
            train_set = smallNORBViewPoint(data_dir, exp=exp, train=True, download=True,
                    transform=transforms.Compose(trans))
            trans = trans[:1] + [transforms.CenterCrop(32)]  +trans[3:]
            valid_set = smallNORBViewPoint(data_dir, exp=exp, train=False, familiar=False, download=False,
                    transform=transforms.Compose(trans))
        elif exp == "full":
            dataset = smallNORB(data_dir, train=True, download=True,
                    transform = transforms.Compose(trans))

    if dataset != "smallnorb":
        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx = indices[split:]
        valid_idx = indices[:split]

        train_set = Subset(dataset, train_idx)
        valid_set = Subset(dataset, valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, valid_loader

def get_test_loader(data_dir,
                    dataset,
                    batch_size,
                    exp='azimuth', # smallnorb only
                    familiar=True, # smallnorb only
                    num_workers=0,
                    pin_memory=False):

    data_dir = data_dir + '/' + dataset

    if dataset == "cifar10":
        trans = [transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        dataset = datasets.CIFAR10(data_dir, train=False, download=False,
                transform=transforms.Compose(trans))

    elif dataset == "svhn":
        normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                     std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
        trans = [transforms.ToTensor(),
                 normalize]
        dataset = datasets.SVHN(data_dir, split='test', download=True,
                transform=transforms.Compose(trans))

    elif dataset == "smallnorb":
        trans = [transforms.Resize(48),
                 transforms.CenterCrop(32),
                 transforms.ToTensor(),
                 #transforms.Normalize((0.7199,), (0.117,))
                 ]
        if exp in VIEWPOINT_EXPS:
            dataset = smallNORBViewPoint(data_dir, exp=exp, familiar=familiar, train=False, download=True,
                                transform=transforms.Compose(trans))
        elif exp == "full":
            dataset = smallNORB(data_dir, train=False, download=True,
                                transform=transforms.Compose(trans))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader





# def get_mnist_loaders(batch_size=128):
#     train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#     test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
#     train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
#     return train_loader, test_loader

# def get_svhn_loaders(batch_size=128):
#     transform = torchvision.transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize((0.1307,), (0.3081,))
# ])
#     train_set = torchvision.datasets.SVHN(root='./data_svhn', split='train', download=True, transform=transform)
#     test_set = torchvision.datasets.SVHN(root='./data_svhn', split='test', download=True, transform=transform)
#     train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
#     return train_loader, test_loader

# def get_smallnorb_loaders(batch_size=128):
#     train_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Grayscale(num_output_channels=3),
#     transforms.ColorJitter(brightness=0.5, contrast=0.5),
#     transforms.Normalize(mean=[0.5], std=[0.5])

    

#     # Create the PyTorch-compatible dataset
#     train_dataset = SmallNORBWrapper(dataset, split='train', transform=train_transform)
#     test_dataset = SmallNORBWrapper(dataset, split='test', transform=test_transform)


#     BATCH_SIZE = 256

#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
#     test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
#     return train_loader, test_loader
# ])