from torch import nn
import torch

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LeakyReLU())
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        output = self.input_layer(input)
        output, hidden = self.gru(output, hidden)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.input_layer = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LeakyReLU())
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, input, input_present, neighbourhood, hidden):
        combined_input = torch.cat((torch.cat((input, input_present), dim=1), neighbourhood), dim=1).view(input.size(0), 1, self.input_size)
        output = self.input_layer(combined_input)
        output, hidden = self.gru(output, hidden)
        output = self.output_layer(output)
        return output, hidden

class NeighbourhoodFullyConnected(nn.Module):
    def __init__(self, neighbourhood_rel_input_size, neighbourhood_load_input_size, hidden_size, output_size):
        super(NeighbourhoodFullyConnected, self).__init__()
        self.output_size = output_size
        self.neighbourhood_rel_input_layer = nn.Sequential(nn.Linear(neighbourhood_rel_input_size, hidden_size), nn.LeakyReLU())
        self.neighbourhood_load_input_layer = nn.Sequential(nn.Linear(neighbourhood_load_input_size, hidden_size), nn.LeakyReLU())
        self.neighbourhood_rel_hidden_layer = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())
        self.neighbourhood_load_hidden_layer = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LeakyReLU())

    def forward(self, neighbourhood_rel_input, neighbourhood_load_input):
        neighbourhood_rel_hidden_val = self.neighbourhood_rel_input_layer(neighbourhood_rel_input)
        neighbourhood_load_hidden_val = self.neighbourhood_load_input_layer(neighbourhood_load_input)
        neighbourhood_rel_hidden_output = self.neighbourhood_rel_hidden_layer(neighbourhood_rel_hidden_val)
        neighbourhood_load_hidden_output = self.neighbourhood_load_hidden_layer(neighbourhood_load_hidden_val)
        batch_size = neighbourhood_rel_hidden_output.size(0)
        # Unsure about this
        #output = torch.mul(neighbourhood_rel_hidden_output.view(batch_size * self.output_size),
        #                   neighbourhood_load_hidden_output.view(batch_size *  self.output_size)).view(batch_size, self.output_size)
        output = torch.mul(neighbourhood_rel_hidden_output,
                           neighbourhood_load_hidden_output)
        
        return output

class NeighbourhoodFullyConnectedDecoder(nn.Module):
    def __init__(self, neighbourhood_rel_input_size, hidden_size, output_size):
        super(NeighbourhoodFullyConnectedDecoder, self).__init__()
        self.output_size = output_size
        self.neighbourhood_rel_input_layer = nn.Sequential(nn.Linear(neighbourhood_rel_input_size, hidden_size), nn.LeakyReLU())
        self.neighbourhood_rel_hidden_layer = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LeakyReLU())

    def forward(self, neighbourhood_rel_input):
        neighbourhood_rel_hidden_val = self.neighbourhood_rel_input_layer(neighbourhood_rel_input)
        neighbourhood_rel_hidden_output = self.neighbourhood_rel_hidden_layer(neighbourhood_rel_hidden_val)
        output = neighbourhood_rel_hidden_output
        return output
