import torch
import torchvision
from torch.utils.data import DataLoader
from diffusers import UNet2DModel,DDIMScheduler,DiffusionPipeline

def load_model(repo_id="google/ddpm-cifar10-32"):
    model = (UNet2DModel.from_pretrained(repo_id))

    return model

def load_dataset(dataset='MNIST', batch_size=128, training=True):
    # Load the MNIST dataset
    if dataset == "MNIST":
        dataset = torchvision.datasets.MNIST(root="mnist/", train=training, download=True,
                                             transform=torchvision.transforms.ToTensor())
    elif dataset == "FASHION":
        dataset = torchvision.datasets.FashionMNIST(root="FashionMNIST/", train=training, download=True,
                                                    transform=torchvision.transforms.ToTensor())

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=training)

    return dataloader

