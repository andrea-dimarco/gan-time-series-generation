from torch import nn, Tensor, zeros
from torch.nn.init import normal_


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, var=0.02, num_layers=3, module_type='gru') -> None:
        '''
        The Generator takes a sequence of iid samples and generates a sequence in the latent space.
        Args:
            - module_type: what module to use between RNN, GRU and LSTM
            - input_size: dimensionality of one sample
            - hidden_size: dimensionality of the sample returned by the module
            - num_layers: depth of the module
            - output_size: size of the final output
            - seq_len: length of the sequence to embed
            - device: which device should the model run on
        '''
        assert(module_type in ['rnn', 'gru', 'lstm'])

        super().__init__()
        self.module_type = module_type
        self.num_layers = num_layers
        self.num_final_layers = int(num_layers/3+1)
        self.hidden_size = hidden_size
        self.output_size = output_size
        
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

        # initialize weights
        for layer_p in self.module._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    normal_(self.module.__getattr__(p), 0.0, var)
        for layer_p in self.final._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    normal_(self.final.__getattr__(p), 0.0, var)


    def forward(self, x: Tensor) -> Tensor:
        '''
        Forward pass
        '''
        batch_size = x.size()[0]
        h0 = zeros(self.num_layers, batch_size, self.hidden_size) # initial state
        h0_final = zeros(self.num_final_layers, batch_size, self.output_size) # initial state

        if self.module_type == 'lstm':
            c0 = zeros(self.num_layers, batch_size, self.hidden_size)
            out, _ = self.module(x, (c0, h0)) # shape = ( batch_size, seq_len, hidden_size )
            out, _ = self.final(out, (c0, h0_final)) # shape = ( batch_size, seq_len, hidden_size )
        else:
            out, _ = self.module(x, h0) # shape = ( batch_size, seq_len, hidden_size )
            out, _ = self.final(out, h0_final) # shape = ( batch_size, seq_len, hidden_size )

        #out = self.norm(out)
        return out