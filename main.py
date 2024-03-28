import argparse
from datetime import datetime
from train import main
import os
import shutil
 
archs = {
    'LeNet5': [('C', 6, 5, 'not_same', 3),
                ('M',2,2),
                ('C', 16, 5, 'not_same'),
                ('M',2,2),
                ('fc' , 400 , 120 , ),
                 ('fc' , 120 , 84),
                ('fc3' , 84 , 10)] ,

    'VGG8': [('C', 128, 3, 'same', 3),
                ('M',2,2),
                ('C', 256, 3, 'same', 1.0),
                ('M',2,2),
                ('C', 512, 3, 'same', 1.0),
                ('M',2,2),
                ('fc' , 8192 , 1024),
                ('fc' , 1024 , 10)],
    # 'VGG11': [('C', 128, 3, 'same', 3),
    #             ('M',2,2),
    #             ('C', 128, 3, 'same', 1.0),
    #             ('M',2,2),
    #             ('C', 256, 3, 'same', 1.0),
    #             ('M',2,2),
    #             ('fc' , 1024 , 256),
    #              ('fc' , 256 , 128),
    #             ('fc3' , 128 , 10)]
}

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
    parser.add_argument('--type', default='cifar10', help='dataset for training')
    parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training (default: 200)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 32)')
    parser.add_argument('--grad_scale', type=float, default=1, help='learning rate for wage delta calculation')
    parser.add_argument('--seed', type=int, default=117, help='random seed (default: 117)')
    parser.add_argument('--log_interval', type=int, default=100,  help='how many batches to wait before logging training status default = 100')
    parser.add_argument('--test_interval', type=int, default=1,  help='how many epochs to wait before another test (default = 1)')
    parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
    parser.add_argument('--decreasing_lr', default='200,250', help='decreasing strategy')
    parser.add_argument('--wl_weight', type = int, default=2)
    parser.add_argument('--wl_grad', type = int, default=8)
    parser.add_argument('--wl_activate', type = int, default=8)
    parser.add_argument('--wl_error', type = int, default=8)
    parser.add_argument('--inference', default=0)
    parser.add_argument('--onoffratio', default=10)
    parser.add_argument('--cellBit', default=1)
    parser.add_argument('--subArray', default=128)
    parser.add_argument('--ADCprecision', default=5)
    parser.add_argument('--vari', default=0)
    parser.add_argument('--t', default=0)
    parser.add_argument('--v', default=0)
    parser.add_argument('--detect', default=0)
    parser.add_argument('--target', default=0)
    parser.add_argument('--nonlinearityLTP', default=0.01)
    parser.add_argument('--nonlinearityLTD', default=-0.01)
    parser.add_argument('--max_level', default=100)
    parser.add_argument('--d2dVari', default=0)
    parser.add_argument('--c2cVari', default=0)
    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    parser,current_time


    for (key, value) in archs.items():
        folder_name = f'Results/{key}/NeuroSIM'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        src = 'NeuroSIM'
        trg = folder_name
        
        files=os.listdir(src)
        for fname in files:
            shutil.copy2(os.path.join(src,fname), trg)
        folder2 = f'Results/{key}/'
        os.chdir(folder2)

        print("#---#"*50 , key) 
        main(parser,current_time, value)
        print("+++||++"*25 , key)
        os.chdir('../..')

 