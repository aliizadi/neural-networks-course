import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import itertools
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# MLP with 2 hidden layers
class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation='relu', hidden_layers_num=2):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        
        self.hidden_layers_num = hidden_layers_num
        self.hidden_layers = []
        for i in range(hidden_layers_num-1):
            self.hidden_layers.append(torch.nn.Linear(hidden_size, hidden_size))

        self.hidden_layers = torch.nn.ModuleList(self.hidden_layers)

        self.fc_last = torch.nn.Linear(hidden_size, output_size)

        self.sigmoid = torch.nn.Sigmoid()
        self.activation = activation

    def forward(self, x):
        if self.activation == 'relu':
            x = torch.nn.functional.relu(self.fc1(x))
            for i in range(self.hidden_layers_num-1):
                x = torch.nn.functional.relu(self.hidden_layers[i](x))

        elif self.activation == 'tanh':
            x = torch.nn.functional.tanh(self.fc1(x))
            for i in range(self.hidden_layers_num-1):
                x = torch.nn.functional.tanh(self.hidden_layers[i](x))
        
        elif self.activation == 'softsign':
            x = torch.nn.functional.softsign(self.fc1(x))
            for i in range(self.hidden_layers_num-1):
                x = torch.nn.functional.softsign(self.hidden_layers[i](x))
        
        elif self.activation == 'softplus':
            x = torch.nn.functional.softplus(self.fc1(x))
            for i in range(self.hidden_layers_num-1):
                x = torch.nn.functional.softplus(self.hidden_layers[i](x))
        
        x = self.fc_last(x)
        return self.sigmoid(x)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_loss_acc(loss_train, loss_val, metrics_train, metrics_val, epochs, learning_rate, batch_size, last_q=False):
    acc_train = [acc[-1] for acc in metrics_train]
    acc_val = [acc[-1] for acc in metrics_val]
    plt.figure(figsize=(15,5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_train, label='train')
    plt.plot(loss_val, label='val')
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.title(f'batch_size={batch_size}\n epochs={epochs}, lr={learning_rate}')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(acc_train, label='train')
    plt.plot(acc_val, label='val')
    plt.xlabel('batch')
    plt.ylabel('accuracy')
    plt.title(f'batch_size={batch_size}\n epochs={epochs}, lr={learning_rate}')
    plt.legend()


    if last_q:
        plt.figure(figsize=(15,5))
        f1_train = [f1[0] for f1 in metrics_train]
        f1_val = [f1[0] for f1 in metrics_val]
        plt.subplot(2, 2, 1)
        plt.plot(f1_train, label='train')
        plt.plot(f1_val, label='val')
        plt.xlabel('batch')
        plt.ylabel('f1')
        plt.title(f'batch_size={batch_size}\n epochs={epochs}, lr={learning_rate}')
        plt.legend()

        precision_train = [prec[1] for prec in metrics_train]
        precision_val = [prec[1] for prec in metrics_val]
        plt.subplot(2, 2, 2)
        plt.plot(precision_train, label='train')
        plt.plot(precision_val, label='val')
        plt.xlabel('batch')
        plt.ylabel('precision')
        plt.title(f'batch_size={batch_size}\n epochs={epochs}, lr={learning_rate}')

        recall_train = [recall[2] for recall in metrics_train]
        recall_val = [recall[2] for recall in metrics_val]
        plt.subplot(2, 2, 3)
        plt.plot(recall_train, label='train')
        plt.plot(recall_val, label='val')
        plt.xlabel('batch')
        plt.ylabel('recall')
 

def plot_loss_acc_comparision(comparision_list, comparision_name, loss_vals, acc_vals):
    acc_vals = [acc[-1] for acc in acc_vals]
    plt.figure(figsize=(10,5))
    plt.title(f'train and validation loss and accuracy comparison for: {comparision_name}')

    plt.subplot(1, 2, 1)
    plt.bar([str(c) for c in comparision_list], [acc[-1] for acc in acc_vals])
    plt.xlabel(comparision_name)
    plt.ylabel('valiation accuracy')

    plt.subplot(1, 2, 2)
    plt.bar([str(c) for c in comparision_list], [loss[-1] for loss in loss_vals])
    plt.xlabel(comparision_name)
    plt.ylabel('valiation loss')


def report_metrics(y_pred, y_true):

    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    return f1, precision, recall, accuracy


class Model:
    def __init__(self, input_size, hidden_size, batch_size, activation='relu', loss='BCE', optimizer='Adam', hidden_layers_num=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = optimizer
        self.hidden_layers_num = hidden_layers_num
        self.model = MLP(input_size=self.input_size, hidden_size=self.hidden_size, output_size=1, activation=activation, hidden_layers_num=hidden_layers_num).float()


    
    def train_evaluate(self, train_dataset, val_dataset, test_dataset, epochs=100, learning_rate=0.01):

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        if self.loss == 'BCE':
            criterion = torch.nn.BCELoss()
        
        elif self.loss == 'MSE':
            criterion = torch.nn.MSELoss()

        elif self.loss == 'L1':
            criterion = torch.nn.L1Loss()
        
        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        elif self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        
        elif self.optimizer == 'RMSprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate)
        

        loss_train = []
        metrics_train = []
        loss_val = []
        metrics_val = []

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.view(-1, self.input_size)
                output = self.model(data.float())
                target = target.unsqueeze(1).float()
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                y_true = target.squeeze().detach().numpy()
                y_pred = output.round().squeeze().detach().numpy()

                train_metrics = report_metrics(y_pred, y_true)
                    
                y_true = []
                y_pred = []
                with torch.no_grad():
                    val_loss = 0
                    val_acc = 0
                    for batch_idx, (data, target) in enumerate(val_loader):
                        data = data.view(-1, self.input_size)
                        output = self.model(data)
                        target = target.unsqueeze(1).float()
                        val_loss += criterion(output, target).item()
                        
                        y_true.extend(target.squeeze().detach().numpy().tolist())
                        y_pred.extend(output.round().squeeze().detach().numpy().tolist())

                    val_metrics = report_metrics(y_pred, y_true)

                loss_train.append(loss.item())
                metrics_train.append(train_metrics)

                loss_val.append(val_loss / len(val_loader))
                metrics_val.append(val_metrics)
    
        test_loss = 0
        cm = np.zeros((2, 2))
        y_true = []
        y_pred = []

        with torch.no_grad():
            
            for batch_idx, (data, target) in enumerate(test_loader):
                data = data.view(-1, self.input_size)
                output = self.model(data)
                target = target.unsqueeze(1).float()
                pred = output.round()
                test_loss += criterion(output, target).item()
                y_true.extend(target.squeeze().detach().numpy().tolist())
                y_pred.extend(pred.squeeze().detach().numpy().tolist())
                for i in range(len(pred)):
                    cm[int(target[i].item()), int(pred[i].item())] += 1

            test_metrics = report_metrics(y_pred, y_true)

        test_loss = test_loss / len(test_loader)

        return loss_train, loss_val, metrics_train, metrics_val, test_loss, test_metrics, cm
