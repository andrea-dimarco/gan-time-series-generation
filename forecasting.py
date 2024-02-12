import warnings
warnings.filterwarnings("ignore")


import torch
'''
Utility
'''
def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X).type(torch.float32), torch.tensor(y).type(torch.float32)


def create_dataset_2(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        X.append(feature)
        y.append(feature)
    return torch.tensor(X).type(torch.float32), torch.tensor(y).type(torch.float32)



'''
Dataset Generation
'''
if True:
    from src.data_generation.wiener_process import multi_dim_wiener_process
    data_dim = 1
    n_samples = 1000
    lookback = 10
    train_size = int(n_samples*0.7)

    dataset = multi_dim_wiener_process(p=data_dim, N=n_samples)
    train = dataset[:train_size]
    test = dataset[train_size:]

    X_train, y_train = create_dataset_2(train, lookback=lookback)
    X_test, y_test = create_dataset_2(test, lookback=lookback)

    dataset = torch.from_numpy(dataset).type(torch.float32)
    train = torch.from_numpy(train).type(torch.float32)
    test = torch.from_numpy(test).type(torch.float32)

    print(f"Training Features: {X_train.size()}, Training Targets {y_train.size()}")
    print(f"Testing Features: {X_train.size()}, Testing Targets {y_train.size()}")
    print(f"Shape: ( num_sequences, seq_len, data_dim )")



'''
Define Model
'''
import torch.nn as nn
class Forecaster(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 num_layers=1):
        super().__init__()
        self.lstm = nn.GRU(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size,
                                output_size)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
    

class Cell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 num_layers=1):
        super().__init__()
        self.module = nn.GRU(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        
        self.fc = nn.Linear(in_features=hidden_size,
                            out_features=output_size)
        
    def forward(self, x):
        # embedd
        x, _ = self.module(x)
        x = self.fc(x)
        return x
    

from src.modules.regressor_cell import RegCell
class AE(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size,
                 num_layers=1):
        super().__init__()

        self.Emb = RegCell(input_size=input_size,
                        hidden_size=hidden_dim,
                        output_size=hidden_dim,
                        num_layers=num_layers
                        )
        self.Rec = RegCell(input_size=hidden_dim,
                        hidden_size=output_size,
                        output_size=output_size,
                        num_layers=num_layers
                        )
        
    def forward(self, x:torch.Tensor):
        x = self.Emb(x)
        x = self.Rec(x)
        return x
    

'''
Training
'''
import numpy as np
import torch.optim as optim
import torch.utils.data as data

if True:
    input_size = data_dim
    hidden_size = 50
    output_size = data_dim
    batch_size = 16
    n_epochs = 10
    val_frequency = n_epochs/10

    loss_fn = nn.MSELoss()
    model = AE(input_size=input_size,
                    hidden_dim=hidden_size,
                    output_size=output_size
                    )
    optimizer = optim.Adam(model.parameters())
    train_loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size)
    
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        if epoch % val_frequency != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            y_pred = model(X_test)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))



'''
Validation
'''
import matplotlib.pyplot as plt
if True:
    with torch.no_grad():
        # shift train predictions for plotting
        train_plot = np.ones_like(dataset) * np.nan
        y_pred = model(X_train)
        y_pred = y_pred[:, -1, :]
        train_plot[lookback : train.size()[0]] = model(X_train)[:, -1, :]

        # shift test predictions for plotting
        test_plot = np.ones_like(dataset) * np.nan
        test_plot[train_size+lookback:dataset.size()[0]] = model(X_test)[:, -1, :]
    # plot
    plt.plot(dataset, c='b')
    plt.plot(train_plot, c='r')
    plt.plot(test_plot, c='g')
    plt.show()