import numpy as np
import os
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class Noise:
    def __init__(self, img, noise=None):
        self.noise = noise
        if type(img) != np.ndarray :
            img = np.transpose(  img.numpy() , (1, 2, 0))
        self.img = img

    def make_noise(self):
        if self.noise == 'random':
            return self.random_noise()
        elif self.noise == 'sp':
            return self.sp_noise()
        elif self.noise == 'gauss':
            return self.gauss_noise()
        else:
            return self.img

    def random_noise(self, noise_num=100):
        '''
        添加随机噪点（实际上就是随机在图像上将像素点的灰度值变为255即白色）
        :param image: 需要加噪的图片
        :param noise_num: 添加的噪音点数目，一般是上千级别的
        :return: img_noise
        '''
        img_noise = self.img
        rows, cols, chn = img_noise.shape
        # 加噪声
        for i in range(noise_num):
            x = np.random.randint(0, rows)#随机生成指定范围的整数
            y = np.random.randint(0, cols)
            img_noise[x, y, :] = 255
        return img_noise

    def sp_noise(self, prob=0.1):
        '''
        添加椒盐噪声
        image:原始图片
        prob:噪声比例
        '''
        image = self.img
        img_noise = np.zeros(image.shape,np.uint8)
        thres = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()#随机生成0-1之间的数字
                if rdn < prob:#如果生成的随机数小于噪声比例则将该像素点添加黑点，即椒噪声
                    img_noise[i][j] = 0
                elif rdn > thres:#如果生成的随机数大于（1-噪声比例）则将该像素点添加白点，即盐噪声
                    img_noise[i][j] = 255
                else:
                    img_noise[i][j] = image[i][j]#其他情况像素点不变
        return img_noise

    def gauss_noise(self, mean=0, var=0.001):
        ''' 
            添加高斯噪声
            image:原始图像
            mean : 均值 
            var : 方差,越大，噪声越大
        '''
        image = self.img
        image = np.array(image/255, dtype=float)#将原始图像的像素值进行归一化，除以255使得像素值在0-1之间
        noise = np.random.normal(mean, var ** 0.5, image.shape)#创建一个均值为mean，方差为var呈高斯分布的图像矩阵
        img_noise = image + noise#将噪声和原始图像进行相加得到加噪后的图像
        if img_noise.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        img_noise = np.clip(img_noise, low_clip, 1.0)#clip函数将元素的大小限制在了low_clip和1之间了，小于的用low_clip代替，大于1的用1代替
        img_noise = np.uint8(img_noise*255)#解除归一化，乘以255将加噪后的图像的像素值恢复
        #cv.imshow("gasuss", img_noise)
        return img_noise


class ToyCifar10(Dataset):

    def __init__(self, root, train=True, noise=None):
        super(ToyCifar10, self).__init__()
        self.train = train
        self.root = root
        self.noise = noise

        path_data = self.check_cifar_dataset_exists()

        if self.train == True:
            self.data = torch.load(path_data+'/train_data.pt')
            self.label = torch.load(path_data+'/train_label.pt')
            mean = [self.data[:,0].mean(), self.data[:,1].mean(), self.data[:,2].mean()]
            std = [self.data[:,0].std(), self.data[:,1].std(), self.data[:,2].std()]
            print("train mean and std:", mean, std)
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        else:
            self.data = torch.load(path_data+'/test_data.pt')
            self.label = torch.load(path_data+'/test_label.pt')
            if self.noise is not None:
                for i in range(self.data.size(0)):
                    img = Noise(self.data[i], noise = self.noise).make_noise()
                    self.data[i] = torch.from_numpy( np.transpose(img, (2, 0, 1)) )
            #print(self.data.shape)
            mean = [self.data[:,0].mean(), self.data[:,1].mean(), self.data[:,2].mean()]
            std = [self.data[:,0].std(), self.data[:,1].std(), self.data[:,2].std()]
            print("test mean and std:", mean, std)
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            


    def check_cifar_dataset_exists(self):
        path_data = self.root
        
        flag_train_data = os.path.isfile(path_data + '/cifar/train_data.pt') 
        flag_train_label = os.path.isfile(path_data + '/cifar/train_label.pt') 
        flag_test_data = os.path.isfile(path_data + '/cifar/test_data.pt') 
        flag_test_label = os.path.isfile(path_data + '/cifar/test_label.pt') 
        if flag_train_data==False or flag_train_label==False or flag_test_data==False or flag_test_label==False:
            print('CIFAR dataset missing - downloading...')
            trainset = torchvision.datasets.CIFAR10(root=path_data + '/cifar/temp', train=True,
                                            download=True, transform=transforms.ToTensor())
            testset = torchvision.datasets.CIFAR10(root=path_data + '/cifar/temp', train=False,
                                        download=True, transform=transforms.ToTensor())  
            train_data=torch.Tensor(50000,3,32,32)
            train_label=torch.LongTensor(50000)
            for idx , example in enumerate(trainset):
                train_data[idx]=example[0]
                train_label[idx]=example[1]
            torch.save(train_data,path_data + '/cifar/train_data.pt')
            torch.save(train_label,path_data + '/cifar/train_label.pt') 
            test_data=torch.Tensor(10000,3,32,32)
            test_label=torch.LongTensor(10000)
            for idx , example in enumerate(testset):
                test_data[idx]=example[0]
                test_label[idx]=example[1]
            torch.save(test_data,path_data + '/cifar/test_data.pt')
            torch.save(test_label,path_data + '/cifar/test_label.pt')
        return os.path.join(path_data, 'cifar')


    def __getitem__(self, index):
        import PIL.Image as Image
        label = self.label[index]
        img = np.transpose( self.data[index].numpy(), (1, 2, 0) ) * 255
        img = img.astype( np.uint8 )
        img = Image.fromarray( img ).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return self.data.size(0)