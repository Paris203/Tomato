# import os
# import glob
# import torch
# import matplotlib.pyplot as plt
# import csv
# from tqdm import tqdm
# from config import max_checkpoint_num, proposalN, eval_trainset, set
# from utils.eval_model import eval

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def load_checkpoint(checkpoint_path, model):
#     """
#     Loads the model parameters from a checkpoint file and prints the keys in the checkpoint.
#     """
#     if os.path.isfile(checkpoint_path):
#         #print(f"Loading checkpoint '{checkpoint_path}'")
#         checkpoint = torch.load(checkpoint_path, weights_only=True)


#         # Load model state dictionary
#         model.load_state_dict(checkpoint['model_state_dict'])
#         print("Model loaded successfully.")
        
        
#         epoch = checkpoint['epoch']
#         learning_rate = checkpoint['learning_rate']
#         train_accuracy = checkpoint.get('train_accuracy', None)

#         print(f"Checkpoint loaded from epoch {epoch}, with learning rate {learning_rate} and train accuracy {train_accuracy:.3f}")
        
#         return epoch, learning_rate, train_accuracy
#     else:
#         raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")



# def create_directories(save_path, checkpoint_path):
#     """Create directories if they do not exist."""
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#         print(f"Created directory for saving metrics: {save_path}")
    
#     if not os.path.exists(checkpoint_path):
#         os.makedirs(checkpoint_path)
#         print(f"Created directory for saving checkpoints: {checkpoint_path}")

# def plot_metrics(metrics, save_path, epoch, phase):
#     """Plots the training/testing metrics."""
#     for metric_name, values in metrics.items():
#         # Ensure the length of x (epochs) matches the length of y (metric values)
#         epochs = range(1, len(values) + 1)

#         plt.figure()
#         plt.plot(epochs, values)  # Use 'epochs' instead of 'range(1, epoch + 1)'
#         plt.title(f'{phase} {metric_name} over epochs')
#         plt.xlabel('Epoch')
#         plt.ylabel(metric_name)
#         plt.savefig(os.path.join(save_path, f'{phase}_{metric_name}_epoch_{epoch}.png'))
#         plt.close()


# def save_accuracies(epoch, train_accuracy, test_accuracy, save_path):
#     """Saves train and test accuracies to a CSV file."""
#     accuracy_file = os.path.join(save_path, 'accuracies.csv')
#     file_exists = os.path.isfile(accuracy_file)

#     with open(accuracy_file, mode='a') as file:
#         writer = csv.writer(file)
#         if not file_exists:
#             writer.writerow(['Epoch', 'Train Accuracy', 'Test Accuracy'])  # Header row
#         writer.writerow([epoch, train_accuracy, test_accuracy])

# def train(model,
#           trainloader,
#           testloader,
#           criterion,
#           optimizer,
#           scheduler,
#           save_path,        # Folder for metrics
#           checkpoint_path,  # Folder for checkpoints
#           start_epoch,
#           end_epoch,
#           save_interval,
#          load_checkpoint_path):

#     # Create directories for saving files
#     create_directories(save_path, checkpoint_path)

#     train_metrics = {
#         'learning_rate': [],
#         'raw_accuracy': [],
#         'local_accuracy': [],
#         'raw_loss_avg': [],
#         'local_loss_avg': [],
#         'windowscls_loss_avg': [],
#         'total_loss_avg': []
#     }

#     test_metrics = {
#         'raw_accuracy': [],
#         'local_accuracy': [],
#         'raw_loss_avg': [],
#         'local_loss_avg': [],
#         'windowscls_loss_avg': [],
#         'total_loss_avg': []
#     }
#     # Load checkpoint if provided
#     if load_checkpoint_path:
#         start_epoch, lr, train_acc = load_checkpoint(load_checkpoint_path, model)
#         print(f"Resuming training from epoch {start_epoch + 1}")
#         # print(f"Checkpoint loaded from epoch {start_epoch}, with learning rate {lr:.6f} and train accuracy { train_acc :.4f}")
#     else:
#         print(f"Starting training from scratch at epoch {start_epoch + 1}")
        
#     raw_correct = 0
#     total_samples = 0

#     for epoch in range(start_epoch + 1, end_epoch + 1):
#         model.train()

#         print('Training %d epoch' % epoch)

#         lr = next(iter(optimizer.param_groups))['lr']
#         train_metrics['learning_rate'].append(lr)

#         for i, data in enumerate(tqdm(trainloader)):
#             if set == 'CUB':
#                 images, labels, _, _ = data
#             else:
#                 images, labels = data
#             images, labels = images.to(device), labels.to(device)

#             optimizer.zero_grad()

#             proposalN_windows_score, proposalN_windows_logits, indices, \
#             window_scores, _, raw_logits, local_logits, _ = model(images, epoch, i, 'train')

#             raw_loss = criterion(raw_logits, labels)
#             local_loss = criterion(local_logits, labels)
#             windowscls_loss = criterion(proposalN_windows_logits,
#                                labels.unsqueeze(1).repeat(1, proposalN).view(-1))

#             if epoch < 2:
#                 total_loss = raw_loss
#             else:
#                 total_loss = raw_loss + local_loss + windowscls_loss

#             total_loss.backward()

#             optimizer.step()

#         # Calculate number of correct predictions
#         pred = raw_logits.max(1, keepdim=True)[1]
#         raw_correct += pred.eq(labels.view_as(pred)).sum().item()
#         total_samples += labels.size(0)

#         # Calculate and store training accuracy
#         raw_accuracy = raw_correct / total_samples
#         train_metrics['raw_accuracy'].append(raw_accuracy)
        
#         scheduler.step()

#         # Evaluation every epoch
#         if eval_trainset:
#             raw_loss_avg, windowscls_loss_avg, total_loss_avg, raw_accuracy, local_accuracy, local_loss_avg\
#                 = eval(model, trainloader, criterion, 'train', save_path, epoch)

#             print(f'Train set: raw accuracy: {100. * raw_accuracy:.2f}%, local accuracy: {100. * local_accuracy:.2f}%')

#             # Store metrics
#             train_metrics['raw_accuracy'].append(raw_accuracy)
#             train_metrics['local_accuracy'].append(local_accuracy)
#             train_metrics['raw_loss_avg'].append(raw_loss_avg)
#             train_metrics['local_loss_avg'].append(local_loss_avg)
#             train_metrics['windowscls_loss_avg'].append(windowscls_loss_avg)
#             train_metrics['total_loss_avg'].append(total_loss_avg)

#             # Plot training metrics
#             plot_metrics(train_metrics, save_path, epoch, 'Train')

#         # Eval test set
#         raw_loss_avg, windowscls_loss_avg, total_loss_avg, raw_accuracy, local_accuracy, \
#         local_loss_avg = eval(model, testloader, criterion, 'test', save_path, epoch)

#         print(f'Test set: raw accuracy: {100. * raw_accuracy:.2f}%, local accuracy: {100. * local_accuracy:.2f}%')

#         # Store metrics
#         test_metrics['raw_accuracy'].append(raw_accuracy)
#         test_metrics['local_accuracy'].append(local_accuracy)
#         test_metrics['raw_loss_avg'].append(raw_loss_avg)
#         test_metrics['local_loss_avg'].append(local_loss_avg)
#         test_metrics['windowscls_loss_avg'].append(windowscls_loss_avg)
#         test_metrics['total_loss_avg'].append(total_loss_avg)

#         # Plot test metrics
#         plot_metrics(test_metrics, checkpoint_path, epoch, 'Test')

#         # Save train and test accuracies
#         save_accuracies(epoch, raw_accuracy, test_metrics['raw_accuracy'][-1], checkpoint_path)

#         # Save checkpoint in the specified checkpoint path
#         if (epoch % save_interval == 0) or (epoch == end_epoch):
#             print('Saving checkpoint')
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'learning_rate': lr,
#                 'train_accuracy': train_metrics['raw_accuracy'][-1],
#             }, os.path.join(checkpoint_path, 'model_checkpoint' + '.pth'))

#         # # Limit the number of checkpoints to max_checkpoint_num
#         # checkpoint_list = [os.path.basename(path) for path in glob.glob(os.path.join(checkpoint_path, '*.pth'))]
#         # if len(checkpoint_list) == max_checkpoint_num + 1:
#         #     idx_list = [int(name.replace('epoch', '').replace('.pth', '')) for name in checkpoint_list]
#         #     min_idx = min(idx_list)
#         #     os.remove(os.path.join(checkpoint_path, 'epoch' + str(min_idx) + '.pth'))

import os
import glob
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np
from config import max_checkpoint_num, proposalN, eval_trainset, set
from utils.eval_model import eval

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_checkpoint(checkpoint_path, model):
    """Loads the model parameters from a checkpoint file."""
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        learning_rate = checkpoint['learning_rate']
        train_accuracy = checkpoint.get('train_accuracy', None)
        print(f"Checkpoint loaded from epoch {epoch}, learning rate {learning_rate}, train accuracy {train_accuracy:.3f}")
        return epoch, learning_rate, train_accuracy
    else:
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

def create_directories(save_path, checkpoint_path):
    """Create directories if they do not exist."""
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

def plot_final_metrics(metrics, save_path, phase):
    """Plots the training/testing metrics over epochs."""
    plt.figure(figsize=(12, 8))
    for metric_name, values in metrics.items():
        plt.plot(range(1, len(values) + 1), values, label=f'{phase} {metric_name}')
    plt.title(f'{phase} Metrics Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'{phase}_final_metrics.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, save_path, phase):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'{phase} Confusion Matrix - Final Epoch')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(save_path, f'{phase}_final_confusion_matrix.png'))
    plt.close()

def save_accuracies(epoch, train_accuracy, test_accuracy, save_path):
    """Saves train and test accuracies to a CSV file."""
    accuracy_file = os.path.join(save_path, 'accuracies.csv')
    file_exists = os.path.isfile(accuracy_file)
    with open(accuracy_file, mode='a') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Epoch', 'Train Accuracy', 'Test Accuracy'])
        writer.writerow([epoch, train_accuracy, test_accuracy])

def train(model, trainloader, testloader, criterion, optimizer, scheduler,
          save_path, checkpoint_path, start_epoch, end_epoch, save_interval, load_checkpoint_path, class_names):
    create_directories(save_path, checkpoint_path)

    train_metrics = {'learning_rate': [], 'raw_accuracy': [], 'local_accuracy': [], 'total_loss_avg': []}
    test_metrics = {'raw_accuracy': [], 'local_accuracy': [], 'total_loss_avg': []}

    all_train_labels, all_train_preds = [], []
    all_test_labels, all_test_preds = [], []

    if load_checkpoint_path:
        start_epoch, lr, train_acc = load_checkpoint(load_checkpoint_path, model)
        print(f"Resuming training from epoch {start_epoch + 1}")
    else:
        print(f"Starting training from scratch at epoch {start_epoch + 1}")

    for epoch in range(start_epoch + 1, end_epoch + 1):
        model.train()
        print(f'Training epoch {epoch}')
        lr = next(iter(optimizer.param_groups))['lr']
        train_metrics['learning_rate'].append(lr)

        total_samples = 0
        raw_correct = 0
        epoch_train_labels, epoch_train_preds = [], []

        for i, data in enumerate(tqdm(trainloader)):
            # images, labels = data
            # images, labels = images.to(device), labels.to(device)
            if set == 'CUB':
                images, labels, _, _ = data
            else:
                images, labels = data
            images, labels = images.to(device), labels.to(device)
            #print(f"images shape is :{images.shape}, label shape :{labels.shape}")

            optimizer.zero_grad()
            proposalN_windows_score, proposalN_windows_logits, indices, \
            window_scores, _, raw_logits, local_logits, _ = model(images, epoch, i, 'train')

            raw_loss = criterion(raw_logits, labels)
            local_loss = criterion(local_logits, labels)
            windowscls_loss = criterion(proposalN_windows_logits,
                               labels.unsqueeze(1).repeat(1, proposalN).view(-1))

            if epoch < 2:
                total_loss = raw_loss
            else:
                total_loss = raw_loss + local_loss + windowscls_loss

            total_loss.backward()

            optimizer.step()


            pred = raw_logits.max(1, keepdim=True)[1]
            raw_correct += pred.eq(labels.view_as(pred)).sum().item()
            total_samples += labels.size(0)

            epoch_train_labels.extend(labels.cpu().numpy())
            epoch_train_preds.extend(pred.cpu().numpy().flatten())

        raw_accuracy = raw_correct / total_samples
        train_metrics['raw_accuracy'].append(raw_accuracy)
        scheduler.step()
        # Evaluation every epoch
        if eval_trainset:
            raw_loss_avg, windowscls_loss_avg, total_loss_avg, test_raw_accuracy, local_accuracy, local_loss_avg\
                = eval(model, trainloader, criterion, 'train', save_path, epoch)

        raw_loss_avg, windowscls_loss_avg, total_loss_avg, test_raw_accuracy, local_accuracy, local_loss_avg = eval(model, trainloader, criterion, 'train', save_path, epoch)
        |#test_metrics['raw_accuracy'].append(test_raw_accuracy)

        # all_test_labels.extend(epoch_test_labels)
        # all_test_preds.extend(epoch_test_preds)

        # Save accuracies for this epoch
        #save_accuracies(epoch, raw_accuracy, test_raw_accuracy, checkpoint_path)

        # Save checkpoint
        if (epoch % save_interval == 0) or (epoch == end_epoch):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'learning_rate': lr,
                'train_accuracy': raw_accuracy,
            }, os.path.join(checkpoint_path, f'model_checkpoint_epoch_{epoch}.pth'))

    # Plot metrics after final epoch
    plot_final_metrics(train_metrics, save_path, 'Train')
    plot_final_metrics(test_metrics, save_path, 'Test')

    # Plot final confusion matrices
    #plot_confusion_matrix(all_train_labels, all_train_preds, class_names, save_path, 'Train')
    #plot_confusion_matrix(all_test_labels, all_test_preds, class_names, save_path, 'Test')

