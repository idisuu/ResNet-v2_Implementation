import json
import glob
import matplotlib.pyplot as plt

import os

def visualize(file_list):
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for block_file in file_list:
        with open(block_file, 'r') as f:
            data = json.load(f)
        name = os.path.basename(block_file).split('.')[0]
        if 'Preactivation' in name:
            color ='red'
        else:
            color='blue'
            
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')    
        
        ax1.plot(data['train_loss'], color=color, label=name + '-train')
        ax1.plot(data['test_loss'], '--', color=color, label=name + '-test')
        ax1.legend()

        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Epoch')
        
        ax2.plot(data['train_acc'], color=color, label=name+'-train')
        ax2.plot(data['test_acc'], '--', color=color, label=name+'-test')
    
        ax2.legend()
        
    return fig1, fig2

json_file_list = glob.glob("result/*")
block_file_list = [file for file in json_file_list if 'Block' in file]
bottleneck_file_list = [file for file in json_file_list if 'Bottleneck' in file]

block_loss, block_acc = visualize(block_file_list)
bottleneck_loss, bottleneck_acc = visualize(bottleneck_file_list)

plt.show()

save_folder = './result'
block_loss.savefig(f"{save_folder}/block_loss.png" )
block_acc.savefig(f"{save_folder}/block_acc.png" )
bottleneck_loss.savefig(f"{save_folder}/bottleneck_loss.png" )
bottleneck_acc.savefig(f"{save_folder}/bottleneck_acc.png" )

