from CNNmodel import ConvNet
from tqdm import tqdm 
import torch
from ShapeData import train_loader, valid_loader

model = ConvNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def LossValidation():
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
    return str(avg_loss)

#TODO: Need to figure out how to implement accuracy correctly currently the size of the tensors are not matching 
def Accuracy():
    num_epochs = 20
    for epoch in range(num_epochs):
        train_correct = 0
        train_total = 0
        valid_correct = 0
        valid_total = 0
        
        for x, y in tqdm(train_loader):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y.to(torch.float32))
            loss.backward()
            optimizer.step()
            
            train_predicted = torch.argmax(output, axis=1)
            train_correct += (train_predicted == y).sum().item()
            train_total += len(y)
            
            valid_predicted = torch.argmax(output, axis=1)
            valid_correct += (valid_predicted == y).sum().item()
            valid_total += len(y)
                
        train_accuracy = train_correct / train_total
        valid_accuracy = valid_correct / valid_total
        
        return str(epoch+1, train_accuracy, valid_accuracy)



print("Validation loss " + LossValidation())
