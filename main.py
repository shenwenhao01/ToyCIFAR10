import torch
from termcolor import colored
import utils
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--root', '-r', type=str, default='./data', help='cifar data root')
parser.add_argument('--resume', action='store_true', help='whether use pretrained model')
parser.add_argument('--type', type=str, default='train')
#parser.add_argument('--exp', type=str, default='default')
parser.add_argument('--noise', required=False, default=None, choices=['random','sp','gauss'],help='add noise to test set')
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--model', '-m', type=str, default='vgg11', help='which model')
parser.add_argument('--epoch', '-e', type=int, default=200, help = 'how many epoch')
args = parser.parse_args()

if args.noise:
    output_dir = os.path.join('outputs',args.model, args.noise)
elif args.lr == 0.05:
    output_dir = os.path.join('outputs',args.model, 'default')
else:
    output_dir = os.path.join('outputs',args.model, 'lr_exp')

def run_train():
    net = utils.build_net(args)
    print(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    trainloader, testloader = utils.load_dataset(args)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD( net.parameters() , lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)
    print(colored("The trained model will be saved in: {}".format(output_dir), "yellow"))
    
    begin_epoch = utils.load_model(net, optimizer, scheduler, output_dir, args.resume)

    best_accuracy = 0.0
    import time
    start=time.time()
    for epoch in range(begin_epoch, args.epoch):
        running_loss=0
        running_error=0
        
        for num_batches, (inputs, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels=labels.to(device)            
            scores=net( inputs ) 
            loss =  criterion( scores , labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.detach().item()
            
            error, _, _ = utils.get_error( scores.detach() , labels, split='train')
            running_error += error.item()      


        # compute stats for the full training set
        total_loss = running_loss/(num_batches+1)
        total_error = running_error/(num_batches+1)
        elapsed = (time.time()-start)/60


        print('epoch=',epoch, '\t time=', elapsed,'min','\t lr=', optimizer.state_dict()['param_groups'][0]['lr'],\
        '\t loss=', total_loss , '\t error=', total_error*100 ,'percent')
        accuracy = utils.eval_on_test_set(testloader, device, net, split='train')
        if accuracy > best_accuracy:
            utils.save_model(net, optimizer, scheduler, output_dir, epoch, best=True)
            best_accuracy = accuracy

        utils.save_model(net, optimizer, scheduler, output_dir, epoch, last=True)
        
        if (epoch + 1) % 100 == 0:
            utils.save_model(net, optimizer, scheduler, output_dir, epoch)

        scheduler.step()
 

def run_test():
    net = utils.build_net(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    _, testloader = utils.load_dataset(args)
    optimizer = torch.optim.SGD( net.parameters() , lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)
    print(colored("Test pretrained model in: {}".format(output_dir), "yellow"))
    
    preepoch = utils.load_model(net, optimizer, scheduler, output_dir, best=True)
    if preepoch:
        utils.eval_on_test_set(testloader, device, net, split='test')
        print("This model has been trained for: {} epochs".format(preepoch-1))
    else:
        print(colored("Pretrained model doesn't exist!", "red"))


if __name__ == '__main__':
    globals()['run_' + args.type]()
