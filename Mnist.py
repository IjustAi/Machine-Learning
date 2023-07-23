import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size=784
hidden_size= 100
num_classes = 10
num_epochs=2
batch_size=100
learning_rate=0.01

train_data=torchvision.datasets.MNIST(root='./data',train=True,
    transform=transforms.ToTensor,download=True)

test_data=torchvision.datasets.MNIST(root='./data',train=False,
    transform=transforms.ToTensor)

train_loader=torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False)

class NeuralNet(nn.Module):
    def __init_(self,input_size,hidden_size,num_classes):
        super(NeuralNet,self).__init__()
        self.l1=nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        self.l2=nn.Linear(hidden_size,num_classes)
    
    def forward(self,x):
        out= self.l1(x)
        out= self.relu(out)
        out = self.l2(out)
        return out 

model = NeuralNet(input_size,hidden_size,num_classes)

loss= nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        images=images.reshape(-1,28*28).to(device)
        labels =labels.to(device)

        outputs= model(images)
        Loss= loss(outputs,labels)

        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()

        if(i+1)%100 ==0:
            print(f"epoch {epoch+1} / {num_epochs}, loss = {loss.item():.4f} ")

with torch.no_grad():
    n_correct=0
    n_samples=0
    for images,labels in test_loader:
        images=images.reshape(-1,28*28).to(device)
        labels =labels.to(device)
        output = model(images)

        _, predictions = torch.max(outputs,1)

        n_samples+=labels.shape[0]
        n_correct+= (predictions == labels).sum().item()

    acc= 100* n_correct/n_samples
    print(f'accuracy={acc}')