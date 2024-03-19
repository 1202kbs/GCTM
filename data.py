import math
import sys
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from couplings import get_coupling


class Sampler:

    def __init__(self, datasets, roots, channels, size, X1_eps_std, coupling, coupling_bs, bs, device='cuda:0'):
        if coupling=='pix2pix' or 'inverse' in coupling:
            if len(datasets) != 1:
                print('\n{} coupling only accepts a single dataset! Aborting...\n')
                sys.exit()
            datasets.append('gaussian') # A dummy dataset which will not be used for sampling
        self.coupling_bs = coupling_bs
        self.channels = channels
        self.size = size
        self.bs = bs
        self.X0_loader = get_img_loader(datasets[0], roots[0], coupling_bs, channels, size)
        self.X1_loader = get_img_loader(datasets[1], roots[1], coupling_bs, channels, size)
        self.coupling = get_coupling(coupling, bs, image_size=size)
        self.X1_eps_std = X1_eps_std
        self.device = device
    
    def __sample_X0__(self):
        return next(iter(self.X0_loader))
    
    def __sample_X1__(self):
        return next(iter(self.X1_loader))
    
    def sample_joint(self):
        X0, X1 = self.__sample_X0__(), self.__sample_X1__()
        X0, X1 = self.coupling(X0,X1)
        X1 = X1 + self.X1_eps_std * torch.randn_like(X1)
        return X0, X1
    
    def sample_X0(self):
        return self.sample_joint()[0]
    
    def sample_X1(self):
        return self.sample_joint()[1]

class TensorDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
        return x

class ImageDataset(datasets.VisionDataset):
    def __init__(self, root, transform=None, **kwargs):
        super().__init__(root, transform=transform)
        
        if not isinstance(root, Path):
            root = Path(root)
        
        self.fpaths = sorted(root.glob('*.png')) + sorted(root.glob('*.jpg'))
        assert len(self.fpaths) > 0, 'No images found in {}'.format(root)
        
    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

def get_img_loader(dataset, root, bs, channels, size):
    img_tfs = transforms.Compose([transforms.ToTensor(),
                                  transforms.Resize(size, antialias=True),
                                  transforms.Normalize(mean=[0.5]*channels, std=[0.5]*channels)])

    NOISE_DATA = ['gaussian', 'sphere']
    batch_size = 1 if dataset in NOISE_DATA else bs
    if dataset == 'gaussian':
        tfs = lambda x : torch.randn(size=[bs,channels,size,size])
        train_data = TensorDataset(torch.zeros(size=[1]), transform=tfs)
        n_samples = 0
    elif dataset == 'sphere':
        tfs = lambda x : F.normalize(torch.randn(size=[bs,channels,size,size]), dim=[1,2,3])
        train_data = TensorDataset(torch.zeros(size=[1]), transform=tfs)
        n_samples = 0
    elif dataset == 'mnist':
        train_data = datasets.MNIST(root=root, train=True, download=False, transform=img_tfs)
        n_samples = 60000
    elif dataset == 'fmnist':
        train_data = datasets.FashionMNIST(root=root, train=True, download=False, transform=img_tfs)
        n_samples = 60000
    elif dataset == 'kmnist':
        train_data = datasets.KMNIST(root=root, train=True, download=True, transform=img_tfs)
        n_samples = 60000
    elif dataset == 'cifar10':
        train_data = datasets.CIFAR10(root=root, train=True, download=False, transform=img_tfs)
        n_samples = 50000
    elif dataset == 'imagenet':
        tfs = transforms.Compose([transforms.ToTensor(),
                                  transforms.Resize(size, antialias=True),
                                  transforms.CenterCrop(size),
                                  transforms.Normalize(mean=[0.5]*channels, std=[0.5]*channels)])
        train_data = datasets.ImageFolder(root=root, transform=tfs)
        n_samples = len(train_data)
    elif dataset == 'ffhq':
        train_data = datasets.ImageFolder(root=os.path.join(root,'FFHQ64','train'), transform=img_tfs)
        n_samples = 69000
    elif dataset == 'ffhq_val':
        train_data = datasets.ImageFolder(root=os.path.join(root,'FFHQ64','val'), transform=img_tfs)
        n_samples = 1000
    elif dataset == 'ffhq_orig_train':
        train_data = datasets.ImageFolder(root=os.path.join(root,'FFHQ','train'), transform=img_tfs)
        n_samples = 69000
        return DataLoader(train_data, batch_size=batch_size)
    elif dataset == 'ffhq_orig_val':
        train_data = datasets.ImageFolder(root=os.path.join(root,'FFHQ','val'), transform=img_tfs)
        n_samples = 69000
        return DataLoader(train_data, batch_size=batch_size)
    elif dataset == 'edges2shoes':
        train_data = datasets.ImageFolder(root=os.path.join(root,'edges2shoes', 'train'), transform=img_tfs)
        n_samples = 69000
    return DataLoader(train_data, batch_size=batch_size, shuffle=True)

def toy_generator(dataset, N):
    if dataset == 'gaussian':
        X = torch.randn(size=[N,2]).float()
    elif dataset == 'mog':
        angles = np.linspace(0, 14 * np.pi / 8, 8)
        X1 = 0.8 * np.cos(angles).reshape(-1,1)
        X2 = 0.8 * np.sin(angles).reshape(-1,1)
        X = np.concatenate([X1,X2], axis=1)
        X = X[:,None] + 0.08 * np.random.normal(size=[X.shape[0],N,X.shape[1]])
        X = X.reshape(8*N,2)
        X = X[np.random.permutation(X.shape[0])[:N]]
        X = 1.5*torch.tensor(X).float()
    elif dataset == 'circles':
        X1 = 0.9 * F.normalize(torch.randn(size=[N,2]))
        X2 = 0.6 * F.normalize(torch.randn(size=[N,2]))
        X3 = 0.3 * F.normalize(torch.randn(size=[N,2]))
        X = torch.cat([X1,X2,X3], dim=0)
        X = X + 0.01 * torch.randn_like(X)
        X = X[torch.randperm(X.shape[0])][:N]
        X = 1.5 * X.float()
    elif dataset == 'circle':
        X = 0.9 * F.normalize(torch.randn(size=[N,2]))
        X = X.float()
    elif dataset == 'checker':
        corners = torch.tensor([[-1,0.5], [0,0.5], [-0.5, 0], [0.5, 0.0], [-1, -0.5], [0, -0.5], [-0.5, -1], [0.5, -1]])
        X = 0.9*torch.cat([corner.reshape(1,2) + 0.5*torch.rand(size=[N,2]) for corner in corners], dim=0).float()
        X = X[torch.randperm(X.shape[0])][:N]
        X = 3.0 * X.float()
    elif dataset == 'grid':
        x = np.linspace(-1, 1, 4)
        y = np.linspace(-1, 1, 4)
        X, Y = np.meshgrid(x, y)
        M = np.concatenate([X.reshape(-1,1),Y.reshape(-1,1)], axis=1) * 0.8
        n = math.ceil(N/M.shape[0])
        S = M[:,None] + 0.05*np.random.normal(size=[M.shape[0],n,2])
        S = S.reshape(-1,2)
        S = S[np.random.permutation(S.shape[0])][:N]
        X = 1.5*torch.tensor(S).float()
    return X

def get_2d_loader(dataset, bs):
    tfs = lambda x : toy_generator(dataset, bs)
    return DataLoader(TensorDataset(torch.zeros(size=[1]), transform=tfs), batch_size=1, shuffle=True)