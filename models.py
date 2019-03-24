from random import random

import torch
from torch import nn


class EncoderRNN(nn.Module):
    def __init__(self, lstm_hidden_size, lstm_layers=2, device="cpu"):
        super(EncoderRNN, self).__init__()
        self.lstm_layers = lstm_layers
        self.device = device
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(
            3, lstm_hidden_size, batch_first=True, num_layers=lstm_layers
        )
        self.output_layer = nn.Linear(lstm_hidden_size, 1)

    def forward(
        self,
        reference_cell_input,
        reference_cell_present_input,
        neighbourhood_input_influence,
        encoder_hidden,
    ):
        assert(len(reference_cell_input.shape) == 2)
        assert(len(reference_cell_present_input.shape) == 2)
        assert(len(neighbourhood_input_influence.shape) == 2)
        batch_size = reference_cell_input.size(0)
        cat_input = torch.cat(
            (reference_cell_input, reference_cell_present_input), dim=1
        )

        encoder_input = torch.cat(
            (cat_input, neighbourhood_input_influence), dim=1
        ).unsqueeze(1)
        output, encoder_hidden = self.lstm(encoder_input, encoder_hidden)
        return output, encoder_hidden

    def init_hidden(self, batch_size):
        return (
            torch.zeros(
                (self.lstm_layers, batch_size, self.lstm_hidden_size),
                device=self.device,
            ),
            torch.zeros(
                (self.lstm_layers, batch_size, self.lstm_hidden_size),
                device=self.device,
            ),
        )


class DecoderRNN(nn.Module):
    def __init__(self, lstm_hidden_size, lstm_layers=2):
        super(DecoderRNN, self).__init__()
        self.lstm_layers = lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(
            18, lstm_hidden_size, batch_first=True, num_layers=lstm_layers
        )
        self.output_layer = nn.Linear(lstm_hidden_size, 1)

    def forward(
        self,
        decoder_input,
        reference_cell_present_target,
        target_neighbourhood_influence,
        decoder_hidden,
    ):
        assert(len(decoder_input.shape) == 2)
        assert(len(reference_cell_present_target.shape) == 2)
        assert(len(target_neighbourhood_influence.shape) == 2)
        batch_size = decoder_input.size(0)
        combined_input = torch.cat(
            (
                torch.cat((decoder_input, reference_cell_present_target), dim=1),
                target_neighbourhood_influence,
            ),
            dim=1,
        )
        output, decoder_hidden = self.lstm(
            combined_input.unsqueeze(1), decoder_hidden
        )
        output = self.output_layer(output.squeeze(1))
        return output.unsqueeze(1), decoder_hidden


class NeighbourhoodInputEmbedding(nn.Module):
    def __init__(self, neighbourhood_hidden_size):
        super(NeighbourhoodInputEmbedding, self).__init__()
        self.neighbourhood_hidden_size = neighbourhood_hidden_size
        self.neighbourhood_rel_input_layer = nn.Sequential(
            nn.Linear(2, neighbourhood_hidden_size), nn.LeakyReLU()
        )
        self.neighbourhood_load_input_layer = nn.Sequential(
            nn.Linear(1, neighbourhood_hidden_size), nn.LeakyReLU()
        )
        self.neighbourhood_rel_hidden_layer = nn.Sequential(
            nn.Linear(neighbourhood_hidden_size, 1), nn.Sigmoid()
        )
        self.neighbourhood_load_hidden_layer = nn.Linear(neighbourhood_hidden_size, 1)

    def forward(self, neighbourhood_rel_input, neighbourhood_load_input):
        assert(len(neighbourhood_rel_input.shape) == 2)
        assert(len(neighbourhood_load_input.shape) == 2)
        neighbourhood_rel_hidden_val = self.neighbourhood_rel_input_layer(
            neighbourhood_rel_input
        )
        neighbourhood_load_hidden_val = self.neighbourhood_load_input_layer(
            neighbourhood_load_input
        )
        neighbourhood_rel_hidden_output = self.neighbourhood_rel_hidden_layer(
            neighbourhood_rel_hidden_val
        )
        neighbourhood_load_hidden_output = self.neighbourhood_load_hidden_layer(
            neighbourhood_load_hidden_val
        )
        output = neighbourhood_rel_hidden_output * neighbourhood_load_hidden_output
        # Learn residual. Subtraction since neighbourhood removes load from reference cell.
        output -= neighbourhood_load_input
        return output


class NeighbourhoodTargetEmbedding(nn.Module):
    def __init__(self, neighbourhood_hidden_size, neighbourhood_output_size):
        super(NeighbourhoodTargetEmbedding, self).__init__()
        self.neighbourhood_hidden_size = neighbourhood_hidden_size
        self.neighbourhood_output_size = neighbourhood_output_size
        self.neighbourhood_rel_input_layer = nn.Sequential(
            nn.Linear(2, neighbourhood_hidden_size), nn.LeakyReLU()
        )
        self.neighbourhood_rel_hidden_layer = nn.Linear(
            neighbourhood_hidden_size, neighbourhood_output_size
        )

    def forward(self, neighbourhood_rel_input):
        assert(len(neighbourhood_rel_input.shape) == 2)
        neighbourhood_rel_hidden_val = self.neighbourhood_rel_input_layer(
            neighbourhood_rel_input
        )
        output = self.neighbourhood_rel_hidden_layer(neighbourhood_rel_hidden_val)
        return output


class DynamicTopologyModel(nn.Module):
    def __init__(
        self,
        neighbourhood_hidden_size=32,
        neighbourhood_cell_count=6,
        neighbourhood_output_size=16,
        lstm_hidden_size=32,
        lstm_layers=2,
        teacher_forcing_probability=1.0,
        device="cpu",
    ):
        super(DynamicTopologyModel, self).__init__()
        self.lstm_layers = lstm_layers
        self.device = device
        self.neighbourhood_cell_count = neighbourhood_cell_count
        self.neighbourhood_output_size = neighbourhood_output_size
        self.neighbourhood_input_embedding = NeighbourhoodInputEmbedding(
            neighbourhood_hidden_size
        )
        self.encoder = EncoderRNN(
            lstm_hidden_size=lstm_hidden_size, lstm_layers=lstm_layers, device=device
        )
        self.neighbourhood_target_embedding = NeighbourhoodTargetEmbedding(
            neighbourhood_hidden_size, neighbourhood_output_size
        )
        self.decoder = DecoderRNN(
            lstm_hidden_size=lstm_hidden_size, lstm_layers=lstm_layers
        )
        self.teacher_forcing_probability = teacher_forcing_probability
        self.parameters = (
            list(self.neighbourhood_input_embedding.parameters())
            + list(self.encoder.parameters())
            + list(self.neighbourhood_target_embedding.parameters())
            + list(self.decoder.parameters())
        )
        # self.state_dict = {**self.neighbourhood_input_embedding.state_dict, **self.encoder.state_dict, **self.neighbourhood_target_embedding.state_dict, **self.decoder.state_dict}

    def forward(
        self,
        input_seq_len,
        target_seq_len,
        reference_cell_input,
        reference_cell_present_input,
        neighbourhood_cell_rel_input,
        neighbourhood_cell_load_input,
        reference_cell_target,
        reference_cell_present_target,
        neighbourhood_cell_rel_target,
    ):

        batch_size = reference_cell_input.size(0)
        decoder_output_seq = torch.zeros(
            (batch_size, target_seq_len.data.numpy()[0], 1), device=self.device
        )

        # Iterate through input sequence.
        for input_seq_idx in range(input_seq_len):
            input_neighbourhood_influence = torch.zeros(
                (batch_size, 1), device=self.device
            )
            encoder_hidden = self.encoder.init_hidden(batch_size)

            for neighbourhood_cell_idx in range(self.neighbourhood_cell_count):
                output = self.neighbourhood_input_embedding(
                    neighbourhood_cell_rel_input[
                        :, input_seq_idx, neighbourhood_cell_idx, :
                    ],
                    neighbourhood_cell_load_input[
                        :, input_seq_idx, neighbourhood_cell_idx
                    ],
                )
                input_neighbourhood_influence = torch.add(
                    input_neighbourhood_influence, output
                )
            encoder_output, encoder_hidden = self.encoder(
                reference_cell_input[:, input_seq_idx],
                reference_cell_present_input[:, input_seq_idx],
                input_neighbourhood_influence,
                encoder_hidden,
            )

        # First decoder hidden value should equal encoders hidden value.
        decoder_hidden = encoder_hidden
        # Last input is first input to decoder.
        decoder_input = reference_cell_input[:, -1]

        # Iterate through target sequence.
        for target_seq_idx in range(target_seq_len):
            target_neighbourhood_influence = torch.zeros(
                (batch_size, self.neighbourhood_output_size), device=self.device
            )
            for neighbourhood_cell_idx in range(self.neighbourhood_cell_count):
                output = self.neighbourhood_target_embedding(
                    neighbourhood_cell_rel_target[
                        :, target_seq_idx, neighbourhood_cell_idx, :
                    ]
                )
                target_neighbourhood_influence = torch.add(
                    target_neighbourhood_influence, output
                )
            decoder_output, decoder_hidden = self.decoder(
                decoder_input,
                reference_cell_present_target[:, target_seq_idx],
                target_neighbourhood_influence,
                decoder_hidden,
            )
            decoder_input = (
                decoder_output[:, 0 , :] 
                if self.teacher_forcing_probability > random()
                else reference_cell_target[:, target_seq_idx]
            )
            decoder_output_seq[:, target_seq_idx, :] = decoder_output[:, 0, :]
        return decoder_output_seq


class SimpleEnd2End(nn.Module):
    def __init__(self,):
        super(SimpleEnd2End, self).__init__()
        self.input_layer = nn.Sequential(nn.Linear(32, 64), nn.ReLU())
        self.output_layer = nn.Sequential(nn.Linear(64, 1))

    # def forward(self, input):
    def forward(
        self,
        reference_cell_input,
        reference_cell_present_input,
        neighbourhood_cell_rel_input,
        neighbourhood_cell_load_input,
        neighbourhood_cell_rel_target,
    ):
        temp0 = torch.cat((reference_cell_input, reference_cell_present_input), dim=1)
        temp1 = torch.cat((temp0, neighbourhood_cell_rel_input), dim=1)
        temp2 = torch.cat((temp1, neighbourhood_cell_load_input), dim=1)
        temp3 = torch.cat((temp2, neighbourhood_cell_rel_target), dim=1)
        output = self.input_layer(temp3)
        output = self.output_layer(output)
        return output


class SimpleDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, target_seq_len):
        super(DecoderStupid, self).__init__()
        self.input_size = input_size
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.LeakyReLU()
        )
        self.output_layer = nn.Sequential(nn.Linear(hidden_size, target_seq_len))

    def forward(self, input, input_present, neighbourhood, hidden):
        combined_input = torch.cat(
            (torch.cat((input, input_present), dim=2), neighbourhood), dim=2
        )
        output = self.input_layer(combined_input)
        output = self.output_layer(output)
        return output
