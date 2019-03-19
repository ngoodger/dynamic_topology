from torch import nn
import torch
from random import random

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, gru_layers=2):
        super(EncoderRNN, self).__init__()
        self.gru_layers = gru_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=gru_layers)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, reference_cell_input,reference_cell_present_input, neighbourhood_input_influence, hidden):
        batch_size = reference_cell_input.size(0)
        cat_input = (torch.cat((reference_cell_input[:, seq_idx].view(batch_size, -1),
                                            reference_cell_present_input[:, seq_idx].view(batch_size, -1)), dim=1))
        encoder_input = torch.cat((cat_input , neighbourhood_input_influence), dim=1).view(batch_size,-1)
        output, hidden = self.gru(encoder_input, hidden)
        return output, hidden

    def init_hidden(batch_size):
        return torch.zeros(self.gru_layers, batch_size, self.hidden_size)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, gru_layers=2):
        super(DecoderRNN, self).__init__()
        self.gru_layers = gru_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=gru_layer)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, reference_cell_target, reference_cell_present_target, target_neighbourhood_influence, decoder_hidden):
        combined_input = torch.cat((torch.cat((reference_cell_input, input_present), dim=1), neighbourhood), dim=1)
        batch_size = input.size(0)
        output = self.input_layer(combined_input)
        output, hidden = self.gru(output.view(batch_size, 1, -1), hidden)
        output = self.output_layer(output)
        return output, hidden


class NeighbourhoodInputEmbedding(nn.Module):
    def __init__(self,  neighbourhood_hidden_size):
        super(NeighbourhoodInputEmbedding, self).__init__()
        self.neighbourhood_hidden_size= neighbourhood_hidden_size
        self.neighbourhood_output_size= neighbourhood_output_size
        self.neighbourhood_rel_input_layer = nn.Sequential(nn.Linear(2, neighbourhood_hidden_size), nn.LeakyReLU())
        self.neighbourhood_load_input_layer = nn.Sequential(nn.Linear(1, neighbourhood_hidden_size), nn.LeakyReLU())
        self.neighbourhood_rel_hidden_layer = nn.Sequential(nn.Linear(neighbourhood_hidden_size, 1), nn.Sigmoid())
        self.neighbourhood_load_hidden_layer = nn.Linear(neighbourhood_hidden_size, 1)

    def forward(self, neighbourhood_rel_input, neighbourhood_load_input):
        neighbourhood_rel_hidden_val = self.neighbourhood_rel_input_layer(neighbourhood_rel_input)
        neighbourhood_load_hidden_val = self.neighbourhood_load_input_layer(neighbourhood_load_input)
        neighbourhood_rel_hidden_output = self.neighbourhood_rel_hidden_layer(neighbourhood_rel_hidden_val)
        neighbourhood_load_hidden_output = self.neighbourhood_load_hidden_layer(neighbourhood_load_hidden_val)
        output = neighbourhood_rel_hidden_output * neighbourhood_load_hidden_output
        # Learn residual. Subtraction since neighbourhood removes load from reference cell.
        output -= neighbourhood_load_input 
        return output

class NeighbourhoodTargetEmbedding(nn.Module):
    def __init__(self, neighbourhood_hidden_size, neighbourhood_output_size):
        super(NeighbourhoodTargetEmbedding, self).__init__()
        self.neighbourhood_hidden_size= neighbourhood_hidden_size
        self.neighbourhood_output_size= neighbourhood_output_size
        self.neighbourhood_rel_input_layer = nn.Sequential(nn.Linear(2, neighbourhood_hidden_size), nn.LeakyReLU())
        self.neighbourhood_rel_hidden_layer = nn.Linear(neighbourhood_hidden_size, neighbourhood_output_size)

    def forward(self, neighbourhood_rel_input):
        neighbourhood_rel_hidden_val = self.neighbourhood_rel_input_layer(neighbourhood_rel_input)
        output = self.neighbourhood_rel_hidden_layer(neighbourhood_rel_hidden_val)
        return output

class DynamicTopologyModel():
    def __init__(self, input_size, neighbourhood_hidden_size=32, neighbourhood_cell_count=6, neighbourhood_output_size=16, gru_layers=2, teacher_forcing_probability=1.0):
        self.gru_layers = gru_layers
        self.neighbourhood_cell_count = neighbourhood_cell_count
        self.neighbourhood_input_embedding = NeighbourhoodEncoderEmbedding(neighbourhood_hidden_size, neighbourhood_output_size)
        self.encoder = EncoderRNN(gru_layers=gru_layers)
        self.neighbourhood_target_embedding = NeighbourhoodDecoderEmbedding(neighbourhood_hidden_size, neighbourhood_output_size)
        self.decoder = DecoderRNN(gru_layers=gru_layers)
        self.teacher_forcing_probability = teacher_forcing_probability 

    def forward(self, input_seq_len, target_seq_len, reference_cell_input, reference_cell_present_input,
                neighbourhood_cell_rel_input, neighbourhood_cell_load_input, neighbourhood_cell_rel_target):

        decoder_output_seq = torch.zeros((batch_size, target_seq_len))
        batch_size = reference_cell_input.size(0)

        # Iterate through input sequence.
        for input_seq_idx in range(input_seq_len):
            input_neighbourhood_influence = torch.zeros(batch_size, neigbourhood_output_size)
            encoder_hidden = encoder.init_hidden(batch_size)

            for neighbourhood_cell_idx in range(self.neighbourhood_cell_count):
                output = neighbourhood_input_embedding(neighbourhood_cell_rel_input[:,input_seq_idx, neighbourhood_cell_idx, :].view(-1, 2),
                                              neighbourhood_cell_load_input[:,input_seq_idx, neighbourhood_cell_idx].view(-1, 1))
                input_neighbourhood_influence = torch.add(input_neighbourhood_influence, output)
            encoder_output, encoder_hidden = self.encoder(reference_cell_input, reference_cell_present_input, input_neighbourhood_influence, encoder_hidden)

        # First decoder hidden value should equal encoders hidden value.
        decoder_hidden = encoder_hidden
        # Last input is first input to decoder.
        decoder_input = reference_cell_input[:, -1].view(-1, 1)

        # Iterate through target sequence.
        for target_seq_idx in range(target_seq_len):
            target_neighbourhood_influence = torch.zeros(batch_size, neighbourhood_output_size)
            for neighbourhood_cell_idx in range(self.neighbourhood_cell_count):
                output = neighbourhood_target_embedding(neighbourhood_cell_rel_target[:,target_seq_idx, neighbourhood_cell_idx, :].view(-1, 2),
                                              neighbourhood_cell_load_target[:,target_seq_idx, neighbourhood_cell_idx].view(-1, 1))
                target_neighbourhood_influence = torch.add(target_neighbourhood_influence, output)
            decoder_output, decoder_hidden = self.decoder(self, ,input_present.view(-1, 1), target_neighbourhood_influence, decoder_hidden)
            decoder_input = decoder_output if self.teacher_forcing_probability > random() else input_cell_rel_target[target_seq_idx + 1]
        decoder_output_seq[:, target_seq_idx] = deocder_output
        return decoder_output_seq
            


class SimpleEnd2End(nn.Module):
    def __init__(self,):
        super(SimpleEnd2End, self).__init__()
        self.input_layer = nn.Sequential(nn.Linear(32, 64), nn.ReLU())
        self.output_layer = nn.Sequential(nn.Linear(64, 1))

    #def forward(self, input):
    def forward(self, reference_cell_input, reference_cell_present_input, neighbourhood_cell_rel_input, neighbourhood_cell_load_input, neighbourhood_cell_rel_target):
        temp0 = torch.cat((reference_cell_input, reference_cell_present_input), dim=1)
        temp1 = torch.cat((temp0,neighbourhood_cell_rel_input), dim=1)
        temp2 = torch.cat((temp1,   neighbourhood_cell_load_input),dim=1)
        temp3 = torch.cat((temp2, neighbourhood_cell_rel_target), dim=1)
        output = self.input_layer(temp3)
        output = self.output_layer(output)
        return output

class SimpleDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, target_seq_len):
        super(DecoderStupid, self).__init__()
        self.input_size = input_size
        self.input_layer = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LeakyReLU())
        self.output_layer = nn.Sequential(nn.Linear(hidden_size, target_seq_len))

    def forward(self, input, input_present, neighbourhood, hidden):
        combined_input = torch.cat((torch.cat((input, input_present), dim=2), neighbourhood), dim=2)
        output = self.input_layer(combined_input)
        output = self.output_layer(output)
        return output

