import matplotlib
import numpy as np
import matplotlib.pyplot as plt


#Load text file 

def load_log(file):
    results = []
    with open(file) as f:
        for line in f:
            #print(line)
            if 'accuracy on test set = ' in line: 
                line = line.split()
                results.append(float(line[-2]))
    return results

def main():
    plt.rcParams['figure.figsize'] = (12, 8)
    shows = {}
    #shows['vgg11'] = load_log('outputs/logs/default_vgg11.out')
    #shows['vgg13'] = load_log('outputs/logs/resume_vgg13.out')
    #shows['vgg13_0.5_norm'] = load_log('outputs/logs/vgg13_0.5_norm.out')
    #shows['vgg13_w/o_norm'] = load_log('outputs/logs/vgg13_wo_norm.out')
    #shows['vgg13_mix_norm'] = load_log('outputs/logs/vgg13_nores_norm.out')
    #shows['vgg13_bn'] = load_log('outputs/logs/default_vgg13_bn.out')
    #shows['random_noise_vgg13_bn'] = load_log('outputs/logs/randnoise_vgg13_bn.out')
    #shows['sp_noise_vgg13_bn'] = load_log('outputs/logs/spnoise_vgg13_bn.out')
    #shows['gauss_noise_vgg13_bn'] = load_log('outputs/logs/gaussnoise_vgg13_bn.out')
    #shows['vgg16'] = load_log('outputs/logs/default_vgg16.out')
    #shows['vgg19'] = load_log('outputs/logs/default_vgg19.out')
    shows['mobilenetv2_lr=0.05'] = load_log('outputs/logs/default_mobilenetv2.out')
    shows['mobilenetv2_lr=0.1'] = load_log('outputs/logs/lr_mobilenetv2.out')
    shows['mobilenetv2_lr=0.01'] = load_log('outputs/logs/mobilenetv2_lr_0.01.out')
    #shows['vgg13_bn'] = load_log('outputs/logs/default_vgg13_bn.out')
    #shows['vgg16_bn'] = load_log('outputs/logs/default_vgg16_bn.out')
    #shows['vgg19_bn'] = load_log('outputs/logs/default_vgg19_bn.out')

    for key in sorted(shows.keys()):
        print(key)
        for i in range(50):
            if (i+1)%20 == 0:
                print(format(shows[key][i],'.2f'))
        epochs = np.arange(1, 1+len(shows[key]))
        plt.plot(epochs, shows[key], label='{:16}{}'.format(key, np.max(shows[key])))
    #plt.legend(shows.keys(), loc='upper left')
    plt.legend(bbox_to_anchor=(0.8, 0.8), loc=0, borderaxespad=0.)
    plt.show()
    plt.savefig('outputs.png')

if __name__ == '__main__':
    main()