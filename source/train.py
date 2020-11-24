import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from source.models import ESPCN
from source.datasets import TrainDataset, EvalDataset
from source.utils import AverageMeter, calc_psnr


def training(dict_train):
    """ Train the model

    Trains the model using training and eval datasets. Output directory contains all weights marked by epoch number. The weights corresponding to the smallest lost value are saved as 'best.pth'. The script displays a progressbar and psnr values for each epoch as its running.

    :param dict_train: dictionary containing all configuration values for training
    :return: None
    
    """

    # Configuration values from input dictionary
    train_file= dict_train['training file']
    eval_file= dict_train['eval file']
    outputs_dir= dict_train['output dir']
    scale= dict_train['scale']
    lr= float(dict_train['lr'])
    batch_size= dict_train['batch size']
    num_epochs= dict_train['number of epochs']
    num_workers= dict_train['number of workers']
    seed= dict_train['seed']

    outputs_dir = os.path.join(outputs_dir, 'x{}'.format(scale))

    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    
    cudnn.benchmark = True
    device = torch.device('cpu')    # OR device = torch.device('cuda:0')

    torch.manual_seed(seed)

    # Define model object, eval criterion and optimizer
    model = ESPCN(scale_factor=scale).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.first_part.parameters()},
        {'params': model.last_part.parameters(), 'lr': lr * 0.1}
    ], lr=lr)

    # Load training data
    train_dataset = TrainDataset(train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)
    # Load eval data
    eval_dataset = EvalDataset(eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    # Iterate over epochs
    for epoch in range(num_epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * (0.1 ** (epoch // int(num_epochs * 0.8)))

        # Training
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

        torch.save(model.state_dict(), os.path.join(outputs_dir, 'epoch_{}.pth'.format(epoch)))
        
        # Evaluation
        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        # Update best psnr value
        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(outputs_dir, 'best.pth'))