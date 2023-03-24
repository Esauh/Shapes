from CNNmodel import ConvNet
from tqdm import tqdm 
import torch
from ShapeData import train_loader, valid_loader

model = ConvNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for x, y in tqdm(train_loader):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y.to(torch.float32))
    loss.backward()
    optimizer.step()

model.eval()
avg_loss = 0
for x, y in valid_loader:
    output = model(x)
    loss = criterion(output.squeeze(1), y.to(torch.float32))
    avg_loss += loss.item()
avg_loss /= len(valid_loader)


print("Validation loss " + str(avg_loss))
