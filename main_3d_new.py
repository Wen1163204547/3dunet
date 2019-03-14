import torch
from torch import nn
from torch import optim
from importlib import import_module
from data_3d import DataLoader3d as DatasetLoader 
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models.sync_batchnorm import patch_replication_callback
from hijack import hijack
import numpy as np
import argparse
import time
import os
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='U-Net 2d')
parser.add_argument('--resume', '-m', metavar='RESUME', default='',
                     help='model parameters to load')
parser.add_argument('--save_dir', default='', type=str, metavar='PATH',
                     help='path to save checkpoint files')
parser.add_argument('--test', default=0, type=int, metavar='TEST',
                     help='1 do test evaluation, 0 not')
parser.add_argument('--batchsize', '-b', default=1, type=int, metavar='BATCHSIZE',
                     help='batch size')

class DiceLoss(nn.Module):
    def __init__(self, batch_size):
        super(DiceLoss, self).__init__()
        self.batch_size = batch_size 

    def forward(self, out, seg):
        b, z, w, h = seg.shape
        seg = seg.unsqueeze(1)
        seg_one_hot = Variable(torch.FloatTensor(b,2, z, w, h)).zero_().cuda()
        seg = seg_one_hot.scatter_(1, seg, 1)
        loss = Variable(torch.FloatTensor(b)).zero_().cuda()
        for i in range(2):
            loss += (1 - 2.*((out[:,i]*seg[:,i]).sum(1).sum(1).sum(1)) / ((out[:,i]*out[:,i]).sum(1).sum(1).sum(1)+(seg[:,i]*seg[:,i]).sum(1).sum(1).sum(1)+1e-15))
        loss = loss.mean() / self.batch_size
        del seg_one_hot, seg
        return loss
    
def main():
    global args
    args = parser.parse_args()
    model = 'models.3d_unet'
    net = import_module(model).get_model()
    loss = DiceLoss(args.batchsize)
    #loss = torch.nn.CrossEntropyLoss()
    #loss = SoftmaxLoss()
    #hijack(net)
    net = net.cuda()
    loss = loss.cuda()
    net = torch.nn.DataParallel(net)
    patch_replication_callback(net)
    if args.resume:
        checkpoint = torch.load(args.resume)
        net.module.load_state_dict(checkpoint['state_dict'])
    train_dataset = DatasetLoader('dataset/preprocess_3d', 
                               'dataset/preprocess_3d') #, random=64)
    val_dataset = DatasetLoader('dataset/preprocess_3d', 
                               'dataset/preprocess_3d', test=True)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size = args.batchsize,
        shuffle = True,
        num_workers = 1,
        pin_memory=True)
    
    val_loader = DataLoader(
        #train_dataset,
        val_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 1,
        pin_memory=True)

    if args.test == 1:
        test(val_loader, net, loss)
        return
    optimizer = optim.Adam(net.parameters(),
                        lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

    def lr_restart(T0, Tcur, base_lr = 1e-3):
        lr_max = base_lr
        lr_min = base_lr * 1e-3
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1+np.cos(Tcur/float(T0) * np.pi))
        return lr

    T0 = 5
    Tcur = 0
    base_lr = 1e-3

    for epoch in range(1, 1000+1):
        print ("epoch", epoch)
        lr = lr_restart(T0, Tcur, base_lr)
        train(train_loader, net, loss, epoch, optimizer, lr, batch_size=12)
        #validate(val_loader, net, loss)

        Tcur = Tcur + 1
        if Tcur > T0:
            Tcur = 0
            T0 = T0 + 10
            base_lr = base_lr * 0.5

def train(train_loader, net, loss, epoch, optimizer, lr, batch_size):
    st = time.time()
    net.train()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    losses = []
    for i, (ct, seg) in enumerate(train_loader):
        seg = (seg > 0.5).long()
        ct = Variable(ct).cuda()
        seg = Variable(seg).cuda()
        out = net(ct)
        loss_out = loss(out, seg)
        optimizer.zero_grad()
        loss_out.backward()
        optimizer.step()
        losses.append(loss_out.data.cpu().numpy())
        del ct, seg, loss_out, out
    if epoch % 10 == 0:
        state_dict = net.module.state_dict()
        for key in state_dict:
            state_dict[key] = state_dict[key].cpu()
        torch.save({
            'epoch': epoch,
            'save_dir': args.save_dir,
            'state_dict': state_dict},
            os.path.join(args.save_dir, 'train_3d_%04d'%epoch+'.ckpt'))

    et = time.time()
    print('train loss %2.4f, time %2.4f' % (np.array(losses).mean()*args.batchsize, et - st))



def test(val_loader, net, loss=None):
    st = time.time()
    net.eval()
    losses = []
    softmax = nn.Softmax(dim=1)
    for i, (ct, seg, name) in enumerate(val_loader):
        out_results = []
        c1, c2 = 0, 0
        seg = (seg > 0.5).long()
        ct = Variable(ct).cuda()
        seg = Variable(seg).cuda()
        out = net(ct)
        loss_out = loss(out, seg)
        losses.append(loss_out.data.cpu().numpy())
        out_v, out_p = torch.max(softmax(out), 1)
        out_results = out_p.data.cpu().numpy()[0]
        out_p = out_p.flatten()
        seg = seg.flatten()
        c1 = 2.0 * (out_p*seg).sum().data.cpu().numpy()
        c2 = (out_p.sum().data.cpu().numpy() + seg.sum().data.cpu().numpy())
        np.save(args.save_dir+'/'+name[0].split('/')[-1], out_results)

        del ct, seg, loss_out, out, out_v, out_p
        print name[0].split('/')[-1]
        c = c1 / (c2 + 1e-14)
        print 'dice score', c


    et = time.time()
    print('test loss %2.4f, time %2.4f' % (np.array(losses).mean(), et - st))

if __name__ == '__main__':
    main()
