import torch
import torch.nn as nn
import os
import json
import numpy as np
from torch.optim import optimizer
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from efficientnet_pytorch import EfficientNet

#LMI modules
from dataset import NordsonDataSet


def save_img_with_pred(im, im_name, pred_cls, conf, path):
    """
    save images with predictions as text in the image
    Arguments:
        im(tensor): the image tensor data
        im_name(str): the image file name
        pred_cls(str): the predicted class name
        conf(float): the confidence level of the predicted class
        path(str): the output path
    """
    import cv2
    if im.is_cuda:
        im = im.cpu()
    im = im.permute(1, 2, 0).numpy().copy()
    im *= 255
    im = im.astype(np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f'pred: {pred_cls} {conf:.2f}'
    cv2.putText(im, text, (20,20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(path, im_name), im)


def train_loop(dataloader, model, loss_fn, optimizer, device):
    """
    training loop
    Arguments:
        dataloader: pytorch dataloader for training
        model: pytorch model
        loss_fn: the loss function to train the model
        optimizer: the optimizer to train the model
        device: pytorch device (cuda or cpu)
    Returns:
        mean_loss(float): the mean loss value
        correct(float): the mean accuracy
    """
    model.train()
    size = len(dataloader.dataset)
    total_loss = 0
    correct = 0
    n_iter = len(dataloader)
    is_cuda = device.type.find('cuda')!=-1
    for batch, (X, y, _) in enumerate(dataloader):
        if is_cuda:
            X = X.to(device)
            y = y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # get the accuracy
        with torch.no_grad():
            correct += (pred.softmax(1).argmax(1) == y).type(torch.float).sum().item()
            total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 2 == 0:
            loss, current = loss.item(), batch * dataloader.batch_size
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    mean_loss = total_loss/n_iter
    correct /= size
    print(f"Train Stats: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {mean_loss:>8f}")
    return mean_loss,correct


def val_loop(dataloader, model, loss_fn, device, save_imgs_path=None):
    """
    validation loop
    Arguments:
        dataloader: pytorch dataloader for validation
        model: pytorch model
        loss_fn: the loss function to train the model
        optimizer: the optimizer to train the model
        device: pytorch device (cuda or cpu)
        save_imgs_path(str): the path to save images, default=None.
    Returns:
        loss(float): the mean loss value
        correct(float): the mean accuracy
    """
    model.eval()
    id_to_class = dataloader.dataset.id_to_class
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss, correct = 0, 0
    is_cuda = device.type.find('cuda')!=-1
    with torch.no_grad():
        for X, y, names in dataloader:
            if is_cuda:
                X = X.to(device)
                y = y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y).item()
            pred = pred.softmax(1)
            correct += (pred.softmax(1).argmax(1) == y).type(torch.float).sum().item()
            if save_imgs_path:
                for i,x in enumerate(X):
                    pred_id = pred[i].argmax().item()
                    pred_cls = id_to_class[pred_id]
                    save_img_with_pred(x,names[i],pred_cls,pred[i].max(),save_imgs_path)

    loss /= num_batches
    correct /= size
    print(f"Val Stats: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {loss:>8f} \n")
    return loss, correct


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_imgs', required=True, help='the path to input images')
    parser.add_argument('--path_out', required=True, help='the path to the trained model and image outputs')
    parser.add_argument('--class_map_file', default="class_map.json", help='[optional] the class map file providing <class: id>, default="class_map.json" in the "--path_imgs" folder')
    parser.add_argument('--in_channels', default=3, type=int, help='the image number of channels, default=3')
    parser.add_argument('--model_name', default='b0', help='[optional] the model name, default="b0"')
    parser.add_argument('--lr', type=float, default=5e-5, help='[optional] the learning rate, default=5e-5')
    parser.add_argument('--batch', type=int, default=8, help='[optional] the batch size, default=8')
    parser.add_argument('--epoch', type=int, default=200, help='[optional] the num of epoch, default=200')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='[optional] the weight decay i.e. l2 regulator, default=1e-5')
    args = vars(parser.parse_args())

    path_data = args['path_imgs']
    path_class_map = os.path.join(path_data, 'class_map.json')
    in_channels = args['in_channels']
    model_name = 'efficientnet-'+args['model_name']
    path_output = args['path_out']
    batch_size = args['batch']
    lr = args['lr']
    weight_decay = args['weight_decay']
    epochs = args['epoch']

    train_tfms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        ])

    val_tfms = transforms.Compose([
        transforms.ToTensor(),
        ])

    #--------------------------------------------------------------------------------------------

    if not os.path.isfile(path_class_map):
        raise Exception(f"Not found the file: {path_class_map}")

    with open(path_class_map) as f:
        class_map = json.load(f)

    train_dataset = NordsonDataSet(path_data, class_map, transform=train_tfms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    val_dataset = NordsonDataSet(path_data, class_map, transform=val_tfms)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    if torch.cuda.is_available():
        device = torch.device('cuda:0') 
        device_name = torch.cuda.get_device_name(device)
        print(f'using GPU: {device_name}')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    num_classes = len(train_dataset.class_to_id)
    im_size = train_dataset.im_size[:2]
    print(f'found num of classes: {num_classes}')
    print(f'found image size: {train_dataset.im_size}')
    model = EfficientNet.from_pretrained(model_name, num_classes=num_classes, image_size=im_size, in_channels=in_channels).to(device)
    loss_fn = nn.CrossEntropyLoss()
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, alpha=0.9, momentum=0.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if not os.path.isdir(path_output):
        os.makedirs(path_output)

    writer = SummaryWriter(flush_secs=5)
    min_loss = float('inf')
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        
        loss,acc = train_loop(train_dataloader, model, loss_fn, optimizer, device)
        writer.add_scalar('Loss/train', loss, t+1)
        writer.add_scalar('Accuracy/train', acc, t+1)
        
        loss,acc = val_loop(val_dataloader, model, loss_fn, device)
        writer.add_scalar('Loss/val', loss, t+1)
        writer.add_scalar('Accuracy/val', acc, t+1)
        if loss<min_loss:
            print('saving the model to path_output\n')
            min_loss = loss
            torch.save(model.state_dict(), os.path.join(path_output,'best.pt'))

    #load best model
    print('loading the best model')
    model.load_state_dict(torch.load(os.path.join(path_output,'best.pt')))
    #save predicted images
    val_loop(val_dataloader, model, loss_fn, device, save_imgs_path=path_output)
    writer.close()

    print("Done!")
