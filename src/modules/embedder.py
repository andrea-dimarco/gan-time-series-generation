
from torch import nn, Tensor, zeros


class Embedder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device='cpu', num_layers=3, module_type='gru') -> None:
        '''
        The Embedder maps the input sequence to a lower dimensionality representation.
        Args:
            - module_type: what module to use between RNN, GRU and LSTM
            - input_size: dimensionality of one sample
            - hidden_size: dimensionality of the sample returned by the module
            - num_layers: depth of the module
            - output_size: size of the final output
            - device: which device should the model run on
        '''
        assert(module_type in ['rnn', 'gru', 'lstm'])

        super().__init__()
        self.module_type = module_type
        self.num_layers = num_layers
        self.num_final_layers = int(num_layers/3+1)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        
        # input.shape = ( batch_size, seq_len, feature_size )
        if self.module_type == 'rnn':
            self.module = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
            self.final  = nn.RNN(hidden_size, output_size, self.num_final_layers, batch_first=True)
        elif self.module_type == 'gru':
            self.module = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            self.final  = nn.GRU(hidden_size, output_size, self.num_final_layers, batch_first=True)
        elif self.module_type == 'lstm':
            self.module = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.final  = nn.LSTM(hidden_size, output_size, self.num_final_layers, batch_first=True)
        else:
            assert(False)

        #self.norm = nn.BatchNorm1d(seq_len)


    def forward(self, x: Tensor) -> Tensor:
        '''
        Forward pass
        '''
        batch_size = x.size()[0]
        h0 = zeros(self.num_layers, batch_size, self.hidden_size).to(self.device) # initial state
        h0_final = zeros(self.num_final_layers, batch_size, self.output_size).to(self.device) # initial state

        if self.module_type == 'lstm':
            c0 = zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
            out, _ = self.module(x, (c0, h0)) # shape = ( batch_size, seq_len, hidden_size )
            out, _ = self.final(out, (c0, h0_final)) # shape = ( batch_size, seq_len, output_size )
        else:
            out, _ = self.module(x, h0) # shape = ( batch_size, seq_len, hidden_size )
            out, _ = self.final(out, h0_final) # shape = ( batch_size, seq_len, output_size )

        #out = self.norm(out)
        return out



## TESTING AREA
'''
import dataset_handling as dh
import numpy as np
from torch.utils.data import DataLoader

np.random.seed(0)
p = 5
N = 10000
seq_len = 10
seq_type = 'wein'
num_epochs = 0
batch_size = 10

device = 'cpu'

dataset = dh.SequenceDataset(p=p, N=N, seq_len=seq_len, seq_type=seq_type)
train_loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
)
test_loader = train_loader

module_type = 'gru'
input_size = p
hidden_size = 4
output_size = 2
num_layers = 1


model = Embedder(module_type=module_type, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size).to(device)
sequence = torch.zeros((2, seq_len, p))
sequence[0] = dataset[0]
sequence[1] = dataset[1]

out = model(sequence)
print(out[1])
'''



# Train the model
'''
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)  
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, batch in enumerate(train_loader):  
        # origin shape: [N, 1, 28, 28]
        # resized: [N, 28, 28]
        sequences = batch.reshape(-1, seq_len, input_size).to(device)
        
        # Forward pass
        outputs = model(batch)
        labels = outputs
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with no_grad():
    n_correct = 0
    n_samples = 0
    for batch in test_loader:
        outputs = model(batch)
        # max returns (value ,index)
        _, predicted = max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
'''