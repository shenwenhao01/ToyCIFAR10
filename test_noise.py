import cv2
import numpy as np

class Noise:
    def __init__(self, img, noise):
        self.noise = noise
        if type(img) != np.ndarray :
            img = np.transpose(  img.numpy() , (1, 2, 0))
        self.img = img

    def make_noise(self):
        if self.noise == 'random':
            return random_noise()
        elif self.noise == 'sp':
            return sp_noise()
        else:
            return gauss_noise()


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

