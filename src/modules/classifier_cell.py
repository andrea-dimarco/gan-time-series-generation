from torch import nn, Tensor


class ClassCell(nn.Module):
    def __init__(self, 
                 input_size:int, hidden_size:int, num_classes:int,
                 num_layers:int=1, module_type:str='gru'
                 ) -> None:
        '''
        The generic Cell maps the input sequence to a lower dimensionality representation.
        Args:
            - `module_type`: what module to use between RNN, GRU and LSTM
            - `input_size`: dimensionality of one sample
            - `hidden_size`: dimensionality of the hidden layers of the cell
            - `num_layers`: depth of the module
            - `num_classes`: number of classes
        '''
        super().__init__()
        
        # input.shape = ( batch_size, seq_len, feature_size )
        if module_type == 'rnn':
            self.module = nn.RNN(input_size=input_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 batch_first=True
                                 )
        elif module_type == 'gru':
            self.module = nn.GRU(input_size=input_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 batch_first=True
                                 )
        elif module_type == 'lstm':
            self.module = nn.LSTM(input_size=input_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 batch_first=True
                                 )
        else:
            raise ValueError("Wrong module_type argument.")

        self.fc = nn.Linear(in_features=hidden_size,
                            out_features=num_classes)
        self.activation = nn.Sigmoid()

        # initialize weights
        for layer_p in self.module._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.xavier_uniform_(self.module.__getattr__(p))
        self.fc.apply(init_weights)


    def forward(self, x: Tensor) -> Tensor:
        '''
        Forward pass
        '''
        x, _ = self.module(x) # shape = ( batch_size, seq_len, xput_size )
        x = x[:,-1,:] # only consider the last output of each sequence
        x = self.fc(x)
        x = self.activation(x)
        return x
    


def init_weights(m):
    '''
    Initialized the weights of the nn.Sequential block
    '''
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias)