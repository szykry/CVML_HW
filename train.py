import torch
import torch.nn as nn
import sys

from torch import optim
from torchvision import datasets

from utils import get_args, plotResults, transformData
from test import testModel
from model import ConvNet

haveCuda = torch.cuda.is_available()

args = get_args()
numEpoch = args.numEpoch
batch_size = args.bSize  # train set: 104027, test set: 10418 -> useful batch sizes: 32 or 64
num_workers = args.num_workers
train_dir = args.train_dir
test_dir = args.test_dir
model_dir = args.model_dir
trainMode = args.trainMode

targets = ['Bump', 'Bumpy road', 'Bus stop', 'Children', 'Crossing (blue)', 'Crossing (red)', 'Cyclists',
           'Danger (other)', 'Dangerous left turn', 'Dangerous right turn', 'Give way', 'Go ahead', 'Go ahead or left',
           'Go ahead or right', 'Go around either way', 'Go around left', 'Go around right', 'Intersection',
           'Limit 100', 'Limit 120', 'Limit 20', 'Limit 30', 'Limit 50', 'Limit 60', 'Limit 70', 'Limit 80',
           'Limit 80 over', 'Limit over', 'Main road', 'Main road over', 'Multiple dangerous turns',
           'Narrow road (left)', 'Narrow road (right)', 'No entry', 'No entry (both directions)', 'No entry (truck)',
           'No stopping', 'No takeover', 'No takeover (truck)', 'No takeover (truck) end', 'No takeover end',
           'No waiting', 'One way road', 'Parking', 'Road works', 'Roundabout', 'Slippery road', 'Stop',
           'Traffic light', 'Train crossing', 'Train crossing (no barrier)', 'Wild animals',
           'Priority', 'Turn left', 'Turn right'
           ]


def train(epoch, trainLoader):
    running_loss = 0.0
    correct = 0.0
    total = 0

    net.train()

    for i, data in enumerate(trainLoader, 0):

        inputs, labels = data
        if haveCuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():

            running_loss += loss.item()  # Accumulate loss
            _, predicted = torch.max(outputs, 1)  # [batch, num_class] -> [batch, 1]
            correct += predicted.eq(labels).sum().item()  # Count how many of the predictions equal the labels
            total += labels.shape[0]  # Accumulate number of total images seen

    tr_loss = running_loss / i
    tr_corr = correct / total * 100
    print("Train epoch %d loss: %.3f correct: %.2f" % (epoch + 1, running_loss / i, tr_corr))

    return tr_loss, tr_corr


def val(epoch, testLoader):
    running_loss = 0.0
    correct = 0.0
    total = 0

    net.eval()

    for i, data in enumerate(testLoader, 0):

        inputs, labels = data
        if haveCuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        with torch.no_grad():

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += predicted.eq(labels).sum().item()
            total += labels.shape[0]

    val_loss = running_loss / i
    val_corr = correct / total * 100
    print("Test epoch %d loss: %.3f correct: %.2f" % (epoch + 1, running_loss / i, val_corr))

    return val_loss, val_corr


if __name__ == '__main__':

    transform, transform_val = transformData()

    train_dataset = datasets.ImageFolder(train_dir, transform)
    test_dataset = datasets.ImageFolder(test_dir, transform_val)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               shuffle=True,
                                               batch_size=batch_size,
                                               num_workers=num_workers
                                               )
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              num_workers=num_workers
                                              )

    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []

    best_acc = 0

    if trainMode:
        torch.manual_seed(1)  # Set pseudo-random generator seeds to make multiple runs comparable
        if haveCuda:
            torch.cuda.manual_seed(1)

        net = ConvNet(4)
        if haveCuda:
            net = net.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, numEpoch, eta_min=1e-2)

        for epoch in range(numEpoch):

            loss, acc = train(epoch, train_loader)
            train_accs.append(acc)
            train_losses.append(loss)

            loss, acc = val(epoch, test_loader)
            val_accs.append(acc)
            val_losses.append(loss)

            scheduler.step()

            if acc > best_acc:
                print("Best Model, Saving")
                best_acc = acc
                torch.save(net, model_dir)

        # Results
        plotResults(numEpoch, train_accs, train_losses, val_accs, val_losses)

    testModel(targets, train_loader, batch_size, model_dir)
