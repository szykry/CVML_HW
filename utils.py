import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms, get_image_backend
from IPython.display import HTML


def get_args():
    """
    Function for handling command line arguments

    :return: parsed   command line arguments
    """
    parser = argparse.ArgumentParser(description='PyTorch A2C')

    parser.add_argument('--train-dir', type=str, default='./trafficSignsHW/trainFULL',
                        help='path of the train set')
    parser.add_argument('--test-dir', type=str, default='./trafficSignsHW/testFULL',
                        help='path of the test set')
    parser.add_argument('--model-dir', type=str, default='./model/pyVision.pth',
                        help='path of the model')
    parser.add_argument('--numEpoch', type=int, default=20, metavar='NUM_EPOCH',
                        help='number of epochs')
    parser.add_argument('--bSize', type=int, default=64, metavar='BATCH_SIZE',
                        help='batch size')
    parser.add_argument('--num-workers', type=int, default=4, metavar='NUM_WORKERS',
                        help='number of workers')
    parser.add_argument('--lr', type=float, default=1e-1, metavar='LR',
                        help='learning rate')

    # Argument parsing
    return parser.parse_args()


def progress(value, max=100):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))


def traffic_loader(path):
    def my_pil_loader(path):
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        except:
            print('fail to load {} using PIL'.format(img))

    if get_image_backend() == 'accimage':
        try:
            return accimage_loader(path)
        except IOError:
            print('fail to load {} using accimage, instead using PIL'.format(path))
            return my_pil_loader(path)
    else:
        return my_pil_loader(path)


def plotResults(num_epoch, train_accs, train_losses, val_accs, val_losses):
    # X coordinate for plotting
    x = np.arange(num_epoch)

    plt.figure(figsize=(20, 10))

    # Train is red, validation is blue
    plt.subplot(1, 2, 1)
    plt.plot(x, train_accs, 'r')
    plt.plot(x, val_accs, 'b')

    plt.subplot(1, 2, 2)
    plt.plot(x, train_losses, 'r')
    plt.plot(x, val_losses, 'b')

    plt.show()


def transformData():
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # random shift
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transform, transform_val
