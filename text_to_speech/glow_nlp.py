import torch
from torch.autograd import Variable

class NN(torch.nn.Module):
    def __init__(self, n_layers, n_channels):
        super(NN, self).__init__()
        self.n_layers = n_layers
        self.n_channels = n_channels
        
        self.in_layers = torch.nn.ModuleList()
        self.res_layers = torch.nn.ModuleList()
        self.skip_layers = torch.nn.ModuleList()
        
        end = torch.nn.Conv1d(n_channels, 2*n_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        for i in range(n_layers):
            in_layer = torch.nn.Conv1d(n_channels, 2*n_channels, 1)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_layer = torch.nn.Conv1d(n_channels, n_channels, 1)
                res_layer = torch.nn.utils.weight_norm(res_layer, name='weight')
                self.res_layers.append(res_layer)

            skip_layer = torch.nn.Conv1d(n_channels, n_channels, 1)
            skip_layer = torch.nn.utils.weight_norm(skip_layer, name='weight')
            self.skip_layers.append(skip_layer)
    
    def forward(self, x):
        for i in range(self.n_layers):
            in_act = self.in_layers[i](x)
            t_act = torch.nn.functional.tanh(in_act[:, :self.n_channels, :])
            s_act = torch.nn.functional.sigmoid(in_act[:, self.n_channels:, :])
            acts = t_act * s_act
            
            if i < self.n_layers - 1:
                res_acts = self.res_layers[i](acts)
                x = res_acts + x
          
            if i == 0:
                output = self.skip_layers[i](acts)
            else:
                output = self.skip_layers[i](acts) + output
        return self.end(output)

class GlowNLP(torch.nn.Module):
    def __init__(self, n_flows, n_layers, n_channels):
        super(GlowNLP, self).__init__()
        self.n_flows = n_flows
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.nn = torch.nn.ModuleList()

        n_half = int(n_channels/2)

        for k in range(n_flows):
            self.nn.append(NN(n_layers, n_half)) 
    

    def forward(self, forward_input):
        """
        forward_input =  batch x 4096
        """
        x = torch.unsqueeze(forward_input, 2)
        n_half = int(x.size(1)/2)
        s_list = []
        for k in range(self.n_flows):
            if k%2 == 0:
                x_0 = x[:,:n_half,:]
                x_1 = x[:,n_half:,:]
            else:
                x_1 = x[:,:n_half,:]
                x_0 = x[:,n_half:,:]

            output = self.nn[k](x_0)
            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            x_1 = torch.exp(s)*x_1 + b
            s_list.append(s)
            
            if k%2 == 0:
                x = torch.cat([x[:,:n_half,:], x_1],1)
            else:
                x = torch.cat([x_1, x[:,n_half:,:]], 1)
        return x, s_list
    
    def infer(self, x): 
        """
        x =  batch x 4096 x 1
        """
        n_half = int(4096/2)
        
        for k in reversed(range(self.n_flows)):
            if k%2 == 0:
                x_0 = x[:,:n_half,:]
                x_1 = x[:,n_half:,:]
            else:
                x_1 = x[:,:n_half,:]
                x_0 = x[:,n_half:,:]
            
            output = self.nn[k](x_0)
            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            x_1 = (x_1 - b)/torch.exp(s)
            if k%2 == 0:
                x = torch.cat([x[:,:n_half,:], x_1],1)
            else:
                x = torch.cat([x_1, x[:,n_half:,:]], 1)
        
        return x.data
