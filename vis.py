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
    shows['vgg11'] = load_log('outputs/logs/default_vgg11.out')
    print(shows)
    #shows['vgg11_bn'] = load_log('log_ha_vgg11')

    #shows['vgg13'] = load_log('log_vgg13')
    #shows['vgg13_half'] = load_log('log_half_vgg13')
    #shows['vgg13_bn'] = load_log('log_vgg13_bn')
    #shows['vgg13_half_bn'] = load_log('log_half_vgg13_bn')
    for key in sorted(shows.keys()):
        print(key)
        epochs = np.arange(1, 1+len(shows[key]))
        plt.plot(epochs, shows[key], label='{}:{}'.format(key, np.max(shows[key])))
    #plt.legend(shows.keys(), loc='upper left')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='best', borderaxespad=0.)
    plt.show()
    plt.savefig('outputs.png')

if __name__ == '__main__':
    main()