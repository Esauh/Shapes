from CNNmodel import ConvNet
from tqdm import tqdm 
import torch
from ShapeData import train_loader, valid_loader
from matplotlib import pyplot as plt
import torch
from torch.autograd import Variable

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


def fit(epoch, model, data_loader, phase='training', volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile = True
    running_loss = 0.0
    running_correct = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile), Variable(target)
        if phase == 'training':
            optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        #running_loss += F.nll_loss(output, target, size_average=False).data[0]
        running_loss += F.nll_loss(output, target, size_average=False).data
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()

    loss = running_loss / len(data_loader.dataset)
    accuracy = 100. * running_correct / len(data_loader.dataset)

    print(f'{phase} loss is {loss} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)} = {accuracy}')
    return loss, accuracy

train_losses , train_accuracy = [],[]
val_losses , val_accuracy = [],[]
for epoch in range(1,20):
    epoch_loss, epoch_accuracy = fit(epoch,model,train_loader,phase='training')
    val_epoch_loss , val_epoch_accuracy = fit(epoch,model,valid_loader,phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

plt.plot(range(1, len(train_losses)+1), train_losses, 'bo', label='training loss')
plt.plot(range(1, len(val_losses)+1), val_losses, 'r', label='validation loss')
plt.legend()
plt.show()

print("VALIDATION LOSS IS " + str(avg_loss))
