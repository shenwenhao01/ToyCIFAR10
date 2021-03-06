import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from dataset import *
from models.vgg import *
from models.mobilenetv2 import *

label = ('airplane', 'automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship','Truck' )

def get_error( scores , labels , split):
    bs=scores.size(0)
    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == labels)
    num_matches=indicator.sum()
    class_num = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    class_accuracy = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    if split == 'test':
        for l in range(10):
            for i in range(bs):
                if labels[i] == l:
                    class_num[l] += 1
                    if labels[i] == predicted_labels[i]:
                        class_accuracy[l] += 1
    return 1-num_matches.float()/bs, class_accuracy, class_num 


def build_net(args):
    net_dict = {
        'vgg11':vgg11(),
        'vgg13':vgg13(),
        'vgg16':vgg16(),
        'vgg19':vgg19(),
        'vgg13_bn':vgg13_bn(),
        'vgg16_bn':vgg16_bn(),
        'vgg19_bn':vgg19_bn(),
        'mobilenetv2':mobilenetv2()
    }
    net = net_dict[args.model]
    return net
    

def load_dataset(args):
    if args.type == "train":
        testset = ToyCifar10(root=args.root, train=False, noise=None)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
        trainset = ToyCifar10(root=args.root, train=True, noise=args.noise)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    else:
        testset = ToyCifar10(root=args.root, train=False, noise=args.noise)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
        trainloader = None
    '''
    path_data = os.path.join(args.root, 'cifar','temp')
    trainset = torchvision.datasets.CIFAR10(root=path_data, train=True,
                                        download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testset = torchvision.datasets.CIFAR10(root=path_data, train=False,
                                    download=True, transform=transforms.ToTensor()) 
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    '''
    return trainloader, testloader


def eval_on_test_set(testloader, device, net, split):

    running_error=0
    class_accu = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    class_total = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}

    for num_batches, (inputs, labels) in enumerate(testloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        scores = net( inputs ) 

        error, class_accuracy, class_num =  get_error( scores , labels, split)
        if split == 'test':
            for k , v in class_accuracy.items() :
                if k in class_accu . keys ( ) :
                    class_accu [ k ] += v
                else :
                    class_accu [ k ] = v
            for k , v in class_num.items() :
                if k in class_total.keys ( ) :
                    class_total [k] += v
                else :
                    class_total [k] = v
        running_error += error.item()

    total_error = running_error/(num_batches+1)
    accuracy = (1 - total_error) * 100
    print( 'accuracy on test set =', accuracy ,'percent')
    print(" ")

    if split == 'test':
        for i in range(10):
            if class_total[i] > 0:
                print('Test Accuracy of %10s: %2d%% (%2d/%2d)' % ( label[i], 100 * class_accu[i] / class_total[i],
                    np.sum(class_accu[i]), np.sum(class_total[i])))
            else:
                print('Test Accuracy of %5s: N/A (no training examples)' % (label[i]))

    return accuracy


def load_model(net,
               optim,
               scheduler,
               model_dir,
               resume=True,
               best=False):
    if not resume:
        os.system('rm -rf {}'.format(model_dir))
    if not os.path.exists(model_dir):
        return 0

    pths = [
        int(pth.split('.')[0]) for pth in os.listdir(model_dir)
        if pth != 'latest.pth' and pth != 'best.pth'
    ]
    if len(pths) == 0 and 'latest.pth' not in os.listdir(model_dir):
        return 0
    if 'latest.pth' in os.listdir(model_dir) and best == False:
        pth = 'latest'
    elif(best == True):
        pth = 'best'
    else:
        pth = max(pths)
    
    print('load model: {}'.format(os.path.join(model_dir, '{}.pth'.format(pth))))
    pretrained_model = torch.load( os.path.join(model_dir, '{}.pth'.format(pth)), 'cpu' )
    net.load_state_dict(pretrained_model['net'])
    optim.load_state_dict(pretrained_model['optim'])
    scheduler.load_state_dict(pretrained_model['scheduler'])
    return pretrained_model['epoch'] + 1


def save_model(net, optim, scheduler, model_dir, epoch, last=False, best=False):
    os.system('mkdir -p {}'.format(model_dir))
    model = {
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch
    }
    if last:
        torch.save(model, os.path.join(model_dir, 'latest.pth'))
    elif best:
        torch.save(model, os.path.join(model_dir, 'best.pth'))
    else:
        torch.save(model, os.path.join(model_dir, '{}.pth'.format(epoch)))
    return 0