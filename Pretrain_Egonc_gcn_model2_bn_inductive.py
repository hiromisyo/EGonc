import argparse
import os
import os.path as osp
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from torch.nn import functional as F
from torch import optim

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import CitationFull
from torch_geometric.nn import GCNConv

from sklearn.metrics import roc_auc_score

from utils import ensure_path, progress_bar
from models.utils import pprint, ensure_path
from torch.distributions import Categorical

import random
from models import GCN_model2
from models import GCN_model3
from models import GCN_model4
from models import GAT_model2
from models import GAT_model3
from models import Graphsage_model2
from models import Graphsage_model3
from models import GCN_model2_act
from models import GCN_model2_bn
from models import GCN_model3_bn
from models import GAT_model2_bn
from models import Graphsage_model2_bn
from sklearn.model_selection import train_test_split

from torch.optim.lr_scheduler import MultiStepLR


device = ''
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

def train(epoch):
    print('\n Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    seq = g.x
    edge_index = g.edge_index
    optimizer.zero_grad()
    outputs = net(seq, edge_index)[train_indices]
    loss = criterion(outputs, new_labels[train_indices].to('cuda')) 
    print(loss.item())
    loss.backward()
    optimizer.step()
    _, predicted = outputs.max(1)
    total = len(train_indices)
    correct = predicted.eq(new_labels[train_indices].to('cuda')).sum().item()
    acc = 100. * correct / total
    print(f'train acc {acc}')

def valid(epoch):
    global best_acc
    net.eval()
    valid_loss = 0
    correct = 0
    total = 0

    seq = g.x
    edge_index = g.edge_index
    outputs = net(seq, edge_index)[valid_indices]
    loss = criterion(outputs, new_labels[valid_indices].to('cuda'))
    _, predicted = outputs.max(1)
    total = len(valid_indices)
    correct = predicted.eq(new_labels[valid_indices].to('cuda')).sum().item()
    if args.shmode == False:
        progress_bar(epoch, len(valid_indices), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'  % (valid_loss/1., 100.* correct / total, correct, total))
    acc = 100. * correct / total
    print(f'valid acc {acc}')
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, osp.join(args.save_path,'ckpt.pth'))
        best_acc = acc

    return 0

def test1(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    seq = g.x
    edge_index = g.edge_index
    outputs = net(seq, edge_index)[test_indices]
    loss = criterion(outputs, new_labels[test_indices].to('cuda'))
    _, predicted = outputs.max(1)
    total = len(test_indices)
    correct = predicted.eq(new_labels[test_indices].to('cuda')).sum().item()
    if args.shmode == False:
        progress_bar(epoch, len(test_indices), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'  % (test_loss/1., 100.* correct / total, correct, total))
    acc = 100. * correct / total    
    print(f'test acc {acc}')

def relabel(args, old_labels):
    old_label = torch.unique(old_labels)
    new_label = deepcopy(old_label)
    seen_label = new_label[:args.known_class]
    unseen_label = new_label[args.known_class:]
    seen_indices = []
    for label in seen_label:
        indices = np.where(old_labels == label)[0]
        indices = list(indices)
        seen_indices = seen_indices + indices
    unseen_indices = []
    for label in unseen_label:
        indices = np.where(old_labels == label)[0]
        indices = list(indices)
        unseen_indices = unseen_indices + indices
    
    seen_train_indices, seen_test_indices = train_test_split(seen_indices, test_size = 1 - args.train_rate)
    train_indices = seen_train_indices
    valid_indices, test_indices = train_test_split(seen_test_indices, test_size = args.valid_rate / (1 - args.train_rate), random_state = args.seed)
    return g.y, seen_label, unseen_label, train_indices, valid_indices, test_indices, unseen_indices

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default = 0.1, type = float)
    parser.add_argument('--resume', '-r', action = 'store_true', help = 'resume from checkpoint')
    parser.add_argument('--model_type', default = 'softmax', type = str)
    parser.add_argument('--backbone', default = 'GCN_model2_bn', type = str)      
    parser.add_argument('--dataset', default = 'cora', type = str)
    # parser.add_argument('--dataset', default = 'citeseer', type = str)
    parser.add_argument('--gpu', default = '2', type = str, help = 'use gpu')
    parser.add_argument('--known_class', default = 6, type = int)
    parser.add_argument('--seed', default = '9', type = int)
    parser.add_argument('--shmode', action = 'store_true')
    parser.add_argument('--ratio1', default = 0.5, type = float)
    parser.add_argument('--train_rate', default = 0.7)
    parser.add_argument('--valid_rate', default = 0.1)
    args = parser.parse_args()
    
    args.seed = 100 # 

    pprint(vars(args))
    devcie = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    start_epoch = 0

    print('==> Preparing data..')
    if args.dataset == 'cora':
        graph = Planetoid(root = '/Egonc/datasets', name = 'Cora')
    elif args.dataset == 'citeseer':
        graph = Planetoid(root = '/Egonc/datasets', name = 'Citeseer')
    g = graph[0]

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    new_labels, seen_label, unseen_label, train_indices, valid_indices, test_indices, unseen_indices  = relabel(args, g.y) 
    g.x[unseen_indices] = 0.
    args.dim_feat = g.x.shape[1]
    print('==> Building model..')
    if args.backbone == 'GCN_model2_bn':
        net = GCN_model2_bn(g.x.shape[1], 512, 128, args.known_class, 0.5)
    net = net.to(device)
    g = g.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = args.lr, momentum = 0.9, weight_decay = 5e-4)

    save_path1 = osp.join('results', 'D{}-M{}-B{}-C-inductive'.format(args.dataset, args.model_type, args.backbone,))
    save_path2 = 'LR{}-K{}-U{}-Seed{}'.format(str(args.lr), seen_label, unseen_label, str(args.seed))
    args.save_path = osp.join(save_path1, save_path2)
    ensure_path(save_path1, remove = False)
    ensure_path(args.save_path, remove = False)

    scheduler = MultiStepLR(optimizer, milestones = [50, 125], gamma = 0.1)
    for epoch in range(start_epoch, start_epoch + 1000): # 200 # 1000
        scheduler.step()
        train(epoch)
        valid(epoch)
        test1(epoch)
        if (epoch + 1)%10 == 0:
            state = {'net': net.state_dict(), 'epoch': epoch,}
            torch.save(state, osp.join(args.save_path, 'Modelof_Epoch' + str(epoch) + '.pth'))
    