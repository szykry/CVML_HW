import torch
import matplotlib.pyplot as plt

from CVML_HW.train import targets, haveCuda, train_loader
from CVML_HW.utils import get_args

args = get_args()
batch_size = args.batch_size
model_dir = args.model_dir


def testModel():
    inputs, labels = next(iter(train_loader))
    if haveCuda:
        inputs, labels = inputs.cuda(), labels.cuda()

    net = torch.load(model_dir)
    net.eval()
    outputs = net(inputs)

    _, predicted = torch.max(outputs, 1)

    mean = torch.Tensor((0.485, 0.456, 0.406)).unsqueeze(1).unsqueeze(1)
    std = torch.Tensor((0.229, 0.224, 0.225)).unsqueeze(1).unsqueeze(1)

    f, axarr = plt.subplots(batch_size//8, 8, figsize=(30, 20))

    for i, (img, pred) in enumerate(zip(inputs, predicted)):
        img_rescaled = img.cpu() * std + mean  # undo the normalization

        name = targets[pred.cpu().item()]

        axarr[i // 8, i % 8].imshow(img_rescaled.permute(1, 2, 0))  # Permute dimensions

        axarr[i // 8, i % 8].set_title(name)

        axarr[i // 8, i % 8].grid(False)

        axarr[i // 8, i % 8].set_xticks([])
        axarr[i // 8, i % 8].set_yticks([])
