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
    def __init__(self, input_size, hidden_size, output_size, activation='linear', hidden_layers_num=0, dropout=False, batch_norm=False):
        super(MLP, self).__init__()
        self.hidden_layers_num = hidden_layers_num
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        if hidden_layers_num == 0:
            self.fc1 = torch.nn.Linear(input_size, output_size)
            if self.batch_norm:
                self.bn1 = torch.nn.BatchNorm1d(output_size)

        else: 
            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            if self.batch_norm:
                self.bn1 = torch.nn.BatchNorm1d(hidden_size)
            
            self.hidden_layers = []
            self.batch_norms = []
            for i in range(hidden_layers_num-1):
                self.hidden_layers.append(torch.nn.Linear(hidden_size, hidden_size))
                if self.batch_norm:
                    self.batch_norms.append(torch.nn.BatchNorm1d(hidden_size))

            self.hidden_layers = torch.nn.ModuleList(self.hidden_layers)
            self.batch_norms = torch.nn.ModuleList(self.batch_norms)

            self.fc_last = torch.nn.Linear(hidden_size, output_size)
            if self.batch_norm:
                self.bn_last = torch.nn.BatchNorm1d(output_size)

        if dropout:
            self.dropout = torch.nn.Dropout(p=0.2)

            

    def forward(self, x):

        if self.hidden_layers_num == 0:
            x = self.fc1(x)

            if self.batch_norm:
                x = self.bn1(x)

            if self.dropout:
                x = self.dropout(x)

            if self.activation == 'linear':
                return x
            elif self.activation == 'relu':
                return torch.nn.functional.relu(x)

        else:   
            if self.activation == 'relu':
                x = torch.nn.functional.relu(self.fc1(x))
                if self.batch_norm:
                    x = self.bn1(x)
                if self.dropout:
                    x = self.dropout(x)
                for i in range(self.hidden_layers_num-1):
                    x = torch.nn.functional.relu(self.hidden_layers[i](x))
                    if self.batch_norm:
                        x = self.batch_norms[i](x)
                    if self.dropout:
                        x = self.dropout(x)

            elif self.activation == 'tanh':
                x = torch.nn.functional.tanh(self.fc1(x))
                if self.batch_norm:
                    x = self.bn1(x)
                if self.dropout:
                    x = self.dropout(x)
                for i in range(self.hidden_layers_num-1):
                    x = torch.nn.functional.tanh(self.hidden_layers[i](x))
                    if self.batch_norm:
                        x = self.batch_norms[i](x)
                    if self.dropout:
                        x = self.dropout(x)
            
            elif self.activation == 'softsign':
                x = torch.nn.functional.softsign(self.fc1(x))
                if self.batch_norm:
                    x = self.bn1(x)
                if self.dropout:
                    x = self.dropout(x)
                for i in range(self.hidden_layers_num-1):
                    x = torch.nn.functional.softsign(self.hidden_layers[i](x))
                    if self.batch_norm:
                        x = self.batch_norms[i](x)
                    if self.dropout:
                        x = self.dropout(x)
            
            elif self.activation == 'softplus':
                x = torch.nn.functional.softplus(self.fc1(x))
                if self.batch_norm:
                    x = self.bn1(x)
                if self.dropout:
                    x = self.dropout(x)
                for i in range(self.hidden_layers_num-1):
                    x = torch.nn.functional.softplus(self.hidden_layers[i](x))
                    if self.batch_norm:
                        x = self.batch_norms[i](x)
                    if self.dropout:
                        x = self.dropout(x)
            
            x = self.fc_last(x)
            if self.batch_norm:
                x = self.bn_last(x)
            if self.dropout:
                x = self.dropout(x)
            return torch.nn.functional.relu(x)


def plot_loss_acc(loss_train, loss_val, title):
    plt.figure()
    plt.plot(loss_train, label='train')
    plt.plot(loss_val, label='val')
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.title(title)
    plt.legend()


def plot_loss_acc_comparision(comparision_list, comparision_name, loss_vals):
    plt.figure()
    plt.barh([str(c) for c in comparision_list], [loss[-1] for loss in loss_vals])
    plt.ylabel(comparision_name)
    plt.xlabel('valiation loss')


def report_metrics(y_pred, y_true):
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    return f1, precision, recall, accuracy


class Model:
    def __init__(self, input_size, hidden_size, batch_size, activation='linear', loss='BCE', optimizer='Adam', hidden_layers_num=0,
                dropout=False, batch_norm=False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = optimizer
        self.hidden_layers_num = hidden_layers_num
        self.model = MLP(input_size=self.input_size, hidden_size=self.hidden_size, output_size=2, activation=activation, hidden_layers_num=hidden_layers_num,
                        dropout=dropout, batch_norm=batch_norm).float()


    
    def train_evaluate(self, train_dataset, val_dataset, test_dataset, epochs=100, learning_rate=0.01):

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        test_criterion = torch.nn.MSELoss()

        if self.loss == 'MSE':
            criterion = torch.nn.MSELoss()

        elif self.loss == 'MAE':
            criterion = torch.nn.L1Loss()
        
        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        elif self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        

        loss_train = []
        loss_val = []

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.view(-1, self.input_size)
                output = self.model(data.float())
                target = target.unsqueeze(1).float()
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                    
                with torch.no_grad():
                    val_loss = 0
                    val_acc = 0
                    for batch_idx, (data, target) in enumerate(val_loader):
                        data = data.view(-1, self.input_size)
                        output = self.model(data)
                        target = target.unsqueeze(1).float()
                        val_loss += test_criterion(output, target).item()
                    
                loss_train.append(loss.item())
                loss_val.append(val_loss / len(val_loader))

        test_loss = 0
        with torch.no_grad():
            
            for batch_idx, (data, target) in enumerate(test_loader):
                data = data.view(-1, self.input_size)
                output = self.model(data)
                target = target.unsqueeze(1).float()
                test_loss += test_criterion(output, target).item()
              
        test_loss = test_loss / len(test_loader)

        return loss_train, loss_val, test_loss
