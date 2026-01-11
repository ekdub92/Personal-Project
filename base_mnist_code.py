import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='.', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

eval_dataset = datasets.MNIST(root='.', train=False, download=True, transform=transform)
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=32, shuffle=False)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 1

model.train()
for epoch in range(epochs):
    for (images, labels) in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()
correct, total = 0, 0
for (images, labels) in eval_loader:
    outputs = model(images)
    probs, indices = torch.max(outputs, 1)
    total += outputs.size(0)
    correct += (indices == labels).sum()

print(f"total: {total}, correct: {correct}")
