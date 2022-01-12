import torch
import torch.nn as nn
import os
import numpy as np
from torch.optim import optimizer
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
            pred = nn.Softmax(dim=1)(pred)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
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
            pred = nn.Softmax(dim=1)(pred)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
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
    path_data = './data/cropped_224x224'
    model_name = 'efficientnet-b0'
    path_output = './trained-inference-models/2022-01-12_224'
    path_save_imgs = './validation/2022-01-12_224'
    path_tensorboard = './runs/2022-01-12'
    image_size = (224,224)
    batch_size = 32
    lr = 5e-5
    weight_decay = 1e-5
    #lr_decay = 0.97 # for every 2.4 epoch
    epochs = 80

    tfms = transforms.Compose([transforms.Resize(image_size),
                           transforms.ToTensor(),
                           ])

    dataset = NordsonDataSet(path_data, transform=tfms)
    num_class = len(dataset.class_to_id)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8)

    if torch.cuda.is_available():
        device = torch.device('cuda:0') 
        device_name = torch.cuda.get_device_name(device)
        print(f'using GPU: {device_name}')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    model = EfficientNet.from_pretrained(model_name, num_classes=num_class).to(device)
    loss_fn = nn.CrossEntropyLoss()
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, alpha=0.9, momentum=0.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if not os.path.isdir(path_output):
        os.makedirs(path_output)
    if not os.path.isdir(path_save_imgs):
        os.makedirs(path_save_imgs)

    writer = SummaryWriter(log_dir=path_tensorboard, flush_secs=5)
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
    val_loop(val_dataloader, model, loss_fn, device, save_imgs_path=path_save_imgs)

    print("Done!")