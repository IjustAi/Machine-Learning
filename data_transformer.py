import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as numpy
import math

class DummyDataset(Dataset):
    def __init__(self,transform=None):
        xy = np.loadtxt()
        self.x =xy[:,1:]
        self.y = xy[:,[0]]
        self.n_shmples = xy.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        sample=self.x[index],self.y[index]

        if self.transform:
            sample =self.transform(sample)
        return sample

    def __len__(self):
        return self.n_shmples

class ToTensor:
    def __call__(self,sample):
        inputs, targets = sample
        return torch.from_numpy(inputs),torch.from_numpy(targets)

composed =torchvision.transforms.Compose([ToTensor()])
dataset=DummyDataset(transform=composed)
dataloader= DataLoader(dataset=dataset,batch_size=4,shuffle=True,num_workers=2)

num_epochs=10
total_samples=len(dataset)
n_iterations=math.ceil(total_samples/4)


for epoch in range(num_epochs):
    for i,(inputs,labels) in enumerate(dataloder):
        if(i+1)%5 ==0:
            print(f'epoch{epoch+1}/{num_epochs}, step{i+1}/{n_iterations}')
