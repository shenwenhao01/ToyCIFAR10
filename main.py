import torch
import torch.nn as nn
import torch.nn.functional as F
from random import randint
import utils
import time
from models import vgg
import torchvision
import torchvision.transforms as transforms


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

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download = True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download = True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False)


# Build the net
net=vgg.vgg19()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#net = nn.DataParallel(net)
net = net.to(device)


def eval_on_test_set():

    running_error=0

    for num_batches, (inputs, labels) in enumerate(testloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        scores = net( inputs ) 

        error = utils.get_error( scores , labels)

        running_error += error.item()

    total_error = running_error/(num_batches+1)
    print( 'error rate on test set =', total_error*100 ,'percent')

def adjust_learning_rate(lr, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = lr * (0.5 ** (epoch // 30))
    return lr

criterion = nn.CrossEntropyLoss()
# vgg19 lr=0.05   vgg11 lr=0.1
optimizer = torch.optim.SGD( net.parameters() , lr=0.05, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
#cheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)


start=time.time()

for epoch in range(0,200):
        
    # set the running quatities to zero at the beginning of the epoch
    running_loss=0
    running_error=0
    
    # set the order in which to visit the image from the training set
    shuffled_indices=torch.randperm(50000)
 
    for num_batches, (inputs, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        
        # send them to the gpu
        inputs = inputs.to(device)
        labels=labels.to(device)
        
        scores=net( inputs ) 

        # Compute the average of the losses of the data points in the minibatch
        loss =  criterion( scores , labels)
        
        # backward pass to compute dL/dU, dL/dV and dL/dW   
        loss.backward()

        # do one step of stochastic gradient descent: U=U-lr(dL/dU), V=V-lr(dL/dU), ...
        optimizer.step()
        

        # START COMPUTING STATS
        
        # add the loss of this batch to the running loss
        running_loss += loss.detach().item()
        
        # compute the error made on this batch and add it to the running error       
        error = utils.get_error( scores.detach() , labels)
        running_error += error.item()      
    
    
    # compute stats for the full training set
    total_loss = running_loss/(num_batches+1)
    total_error = running_error/(num_batches+1)
    elapsed = (time.time()-start)/60
    

    print('epoch=',epoch, '\t time=', elapsed,'min','\t lr=', optimizer.state_dict()['param_groups'][0]['lr'],\
    '\t loss=', total_loss , '\t error=', total_error*100 ,'percent')
    eval_on_test_set() 
    print(' ')
    
    scheduler.step()

def test():
    # choose a picture at random
    idx=randint(0, 10000-1)
    im=test_data[idx]

    # diplay the picture
    utils.show(im)

    # send to device, rescale, and view as a batch of 1 
    im = im.to(device)
    im= (im-mean) / std
    im=im.view(1,3,32,32)
    print(test_label[idx])

    # feed it to the net and display the confidence scores
    scores =  net(im) 
    probs= F.softmax(scores, dim=1)
    utils.show_prob_cifar(probs.cpu())