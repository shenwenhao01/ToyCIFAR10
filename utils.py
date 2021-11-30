import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from test_noise import *
from models.vgg import *
from models.mobilenetv2 import *

label = ('airplane', 'automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship','Truck' )

def get_error( scores , labels ):

    bs=scores.size(0)
    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == labels)
    num_matches=indicator.sum()
    class_num = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    class_accuracy = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    for l in range(10):
        for i in range(bs):
            if labels[i] == l:
                class_num[l] += 1
                if labels[i] == predicted_labels[i]:
                    class_accuracy[l] += 1
    return 1-num_matches.float()/bs, class_accuracy, class_num 

def show(X):
    if X.dim() == 3 and X.size(0) == 3:
        plt.imshow( np.transpose(  X.numpy() , (1, 2, 0))  )
        plt.show()
    elif X.dim() == 2:
        plt.imshow(   X.numpy() , cmap='gray'  )
        plt.show()
    else:
        print('WRONG TENSOR SIZE')

def show_prob_cifar(p):

    p=p.data.squeeze().numpy()

    ft=15
    #p=p.data.squeeze().numpy()
    y_pos = np.arange(len(p))*1.2
    target=2
    width=0.9
    col= 'blue'
    #col='darkgreen'

    plt.rcdefaults()
    fig, ax = plt.subplots()

    # the plot
    ax.barh(y_pos, p, width , align='center', color=col)

    ax.set_xlim([0, 1.3])
    #ax.set_ylim([-0.8, len(p)*1.2-1+0.8])

    # y label
    ax.set_yticks(y_pos)
    ax.set_yticklabels(label, fontsize=ft)
    ax.invert_yaxis()

    # x label
    ax.set_xticklabels([])
    ax.set_xticks([])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(4)


    for i in range(len(p)):
        str_nb="{0:.2f}".format(p[i])
        ax.text( p[i] + 0.05 , y_pos[i] ,str_nb ,
                 horizontalalignment='left', verticalalignment='center',
                 transform=ax.transData, color= col,fontsize=ft)

    plt.show()

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

def add_noise(path_data):
    raise NotImplementedError
    
    import torchvision
    import torchvision.transforms as transforms
    import cv2
    trainset = torchvision.datasets.CIFAR10(root=path_data, train=True,
                                        download=True, transform=transforms.ToTensor())
    testset = torchvision.datasets.CIFAR10(root=path_data, train=False,
                                    download=True, transform=transforms.ToTensor())  
    train_data=torch.Tensor(50000,3,32,32)
    train_label=torch.LongTensor(50000)
    for idx , example in enumerate(trainset):
        train_data[idx]= example[0]
        train_label[idx]=example[1]
    #torch.save(train_data,path_data + 'train_data.pt')
    #torch.save(train_label,path_data + 'train_label.pt') 

    test_data=torch.Tensor(10000,3,32,32)
    test_label=torch.LongTensor(10000)
    for idx , example in enumerate(testset):
        test_data[idx]=Noise(example[0], 'random').make_noise()
        test_label[idx]=example[1]
    #torch.save(test_data[0],path_data + 'test_data[0].pt')
    #torch.save(test_label,path_data + 'test_label.pt')


def load_dataset(args):
    import torchvision
    import torchvision.transforms as transforms
    add_noise(args.root)
    '''
    test_data = torchvision.datasets.CIFAR10(root=args.root, train=False, download = True)
    rgb_mean = []
    rgb_std = []
    for i in range(test_data.size(1) + 1):
        rgb_mean.append( test_data[:,i].mean().item() )
        rgb_std.append( test_data.std() )
    print(rgb_mean, rgb_std)
    '''
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # compute the mean and std for each channel separately!
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.root, train=True, download = True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root=args.root, train=False, download = True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    return trainloader, testloader

def eval_on_test_set(testloader, device, net, split):

    running_error=0
    class_accu = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    class_total = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}

    for num_batches, (inputs, labels) in enumerate(testloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        scores = net( inputs ) 

        error, class_accuracy, class_num =  get_error( scores , labels)
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