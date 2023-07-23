import torch.nn.functional as f
import torchvision
import torchvision.transforms as transforms
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


num_epochs=4
batch_size=4
learning_rate=0.01

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_data=torchvision.datasets.CIFAR10(root='./data',train=True,
    transform=transforms.ToTensor,download=True)

test_data=torchvision.datasets.CIFAR10(root='./data',train=False,
    transform=transforms.ToTensor)


train_loader=torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truch')

class ConNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv1= nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2) 
        self. conv2 = nn.Con2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5, 100)
        self.fc2 = nn.Linear(100,80)
        self.fc3= nn.Linear(80,10)

    def forward(self,x):
        x= self.pool(F.relu(self.conv1(x)))
        x= self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,16*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

model = ConNet().to(device)
loss= nn.CrossEntropyLoss()
optimizer= torch.optim.SGD(model.parameters(),lr=learning_rate)


for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        images=images.to(device)
        labels =labels.to(device)

        outputs= model(images)
        Loss= loss(outputs,labels)

        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()

        if(i+1)%2000 ==0:
            print(f"epoch {epoch+1} / {num_epochs}, loss = {loss.item():.4f} ")

print('Training finished')

with torch.no_grad():
    n_correct=0
    n_samples=0
    n_class_correct=[0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images,labels in test_loader:
        images=images.to(device)
        labels =labels.to(device)
        output = model(images)

        _, predictions = torch.max(outputs,1)

        n_samples+=labels.size(0)
        n_correct+= (predictions == labels).sum().item()
        
        for i in range(batch_size):
            label = lables[1]
            pred= predictions[i]
            if(label==pred):
                n_claass_correct[label]+=1
            n_class_samples[label]+=1

    acc= 100* n_correct/n_samples
    print(f'model of accuracy={acc}')

    for i in range(10):
        acc=100*n_claass_correct[i]/n_class_samples[i]
        print(f'Accuracy of {classes[i]}:{acc}%')