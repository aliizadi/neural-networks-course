import torch
from torch.utils.data import DataLoader

# class AutoEncoder for dimension reduction
class AutoEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(AutoEncoder, self).__init__()
        self.encoder = []
        self.encoder.append(torch.nn.Linear(input_size, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.encoder.append(torch.nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))

        self.encoder = torch.nn.ModuleList(self.encoder)

        
        self.decoder = []
        reversed_hidden_sizes = list(reversed(hidden_sizes)) 
        for i in range(len(reversed_hidden_sizes)-1):
            self.decoder.append(torch.nn.Linear(reversed_hidden_sizes[i], reversed_hidden_sizes[i+1]))

        self.decoder.append(torch.nn.Linear(reversed_hidden_sizes[-1], input_size))
        self.decoder = torch.nn.ModuleList(self.decoder)

       

    def forward(self, x):

        encoded = self.encoder[0](x)

        for i in range(1, len(self.encoder)):
            encoded = self.encoder[i](encoded)

        decoded = self.decoder[0](encoded)

        for i in range(1, len(self.decoder)):
            decoded = self.decoder[i](decoded)
        
        return encoded, decoded


class Model:
    def __init__(self, input_size, hidden_sizes, batch_size, loss='MSE', optimizer='Adam'):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = optimizer
        self.model = AutoEncoder(input_size=self.input_size, hidden_sizes=self.hidden_sizes).float()


    
    def train_evaluate(self, train_dataset, val_dataset, test_dataset, epochs=100, learning_rate=0.01):

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        loss_trains = []
        loss_vals = []

        for epoch in range(epochs):
            for i, (x, y) in enumerate(train_loader):
                x = x.view(-1, self.input_size)
                optimizer.zero_grad()
                encoded, decoded = self.model(x)
                loss = criterion(decoded, x)
                loss.backward()
                optimizer.step()

                loss_trains.append(loss.item())

                loss_val = 0
                with torch.no_grad():
                    for i, (x, y) in enumerate(val_loader):
                        x = x.view(-1, self.input_size)
                        encoded, decoded = self.model(x)
                        loss = criterion(decoded, x)
                        loss_val += loss.item()

                loss_vals.append(loss_val/len(val_loader))

        loss_test = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x = x.view(-1, self.input_size)
                encoded, decoded = self.model(x)
                loss = criterion(decoded, x)
                loss_test += loss.item()
            
        loss_test /= len(test_loader) 

        # encode the train data
        encoded_train = []
        with torch.no_grad():
            for i, (x, y) in enumerate(train_loader):
                x = x.view(-1, self.input_size)
                encoded, decoded = self.model(x)
                encoded_train.append(encoded)
            encoded_train = torch.cat(encoded_train, dim=0)

        encoded_train = encoded_train.detach().numpy()
        
        # encode the val data
        encoded_val = []
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                x = x.view(-1, self.input_size)
                encoded, decoded = self.model(x)
                encoded_val.append(encoded)
            encoded_val = torch.cat(encoded_val, dim=0)
        
        encoded_val = encoded_val.detach().numpy()

        # encode the test data
        encoded_test = []
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x = x.view(-1, self.input_size)
                encoded, decoded = self.model(x)
                encoded_test.append(encoded)
            encoded_test = torch.cat(encoded_test, dim=0)

        encoded_test = encoded_test.detach().numpy()

              

        return loss_trains, loss_vals, loss_test, encoded_train, encoded_val, encoded_test
