#coding=utf-8
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import shutil
import time
from config import num_classes, model_name, model_path, lr_milestones, lr_decay_rate, input_size, \
    root, end_epoch, save_interval, init_lr, batch_size, CUDA_VISIBLE_DEVICES, weight_decay, \
    proposalN, set, channels, load_checkpoint_path
from utils.train_model import train
from utils.read_dataset import read_dataset
from utils.auto_laod_resume import auto_load_resume
from networks.model import MainNet
import sys
import os

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# Add the parent directory of 'utils' and 'datasets' to the Python path
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#
# from utils.read_dataset import read_dataset


os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_names = [
    'Late_blight', 
    'Two-spotted_spider_mite', 
    'Bacterial_spot', 
    'Leaf_Mold', 
    'Target_Spot', 
    'Tomato_mosaic_virus', 
    'healthy', 
    'Early_blight', 
    'Tomato_Yellow_Leaf_Curl_Virus', 
    'Septoria_leaf_spot'
]


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for resnet 18')
    plt.savefig(f'./resnet.png')
    plt.show()

def main():

    #加载数据
    trainloader, testloader = read_dataset(input_size, batch_size, root, set)
    print(f"len of training dataset {len(trainloader)}, test dataset: {len(testloader)}")

    #定义模型
    model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)

    #设置训练参数
    criterion = nn.CrossEntropyLoss()

    parameters = model.parameters()
    checkpoint_path ='result of the model'
    #加载checkpoint
    save_path = os.path.join(model_path, model_name)
    if os.path.exists(save_path):
        start_epoch, lr = auto_load_resume(model, save_path, status='train')
        assert start_epoch < end_epoch
    else:
        os.makedirs(save_path)
        start_epoch = 0
        lr = init_lr

    # define optimizers
    optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)

    model = model.to(device) # 部署在GPU

    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_decay_rate)

    # 保存config参数信息
    time_str = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy('./config.py', os.path.join(save_path, "{}config.py".format(time_str)))

    # 开始训练
    train(model=model,
          trainloader=trainloader,
          testloader=testloader,
          criterion=criterion,
          optimizer=optimizer,
          scheduler=scheduler,
          save_path=save_path,
          checkpoint_path= checkpoint_path,
          start_epoch=start_epoch,
          end_epoch=end_epoch,
          save_interval=save_interval,
         load_checkpoint_path=False,
         )

    # Evaluate the model on the test set
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels =  inputs.to(device), labels.to(device)
            #print(f"inputs shape: {inputs.shape}")
            proposalN_windows_score, proposalN_windows_logits, indices, \
            window_scores, _, raw_logits, local_logits, _  = model(inputs, 1, 6, 'test')
            _, preds = torch.max(raw_logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Print the classification report
    # print(f'\nClassification Report for {model_name}')
    print(classification_report(all_labels, all_preds, zero_division=0))
    
    # # Plot the confusion matrix
    plot_confusion_matrix(all_labels, all_preds)
    #print(all_preds)



if __name__ == '__main__':
    main()
