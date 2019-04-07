
class Seq2Seq(nn.Module):
    def __init__(self, input_size,
                 encoder_hidden_size,
                 decoder_hidden_size,
                 batch_first=False,
                 encoder_num_layers=1,
                 decoder_num_layers=1,
                ):

        super(Seq2Seq, self).__init__()
        self.input_sise = input_size
        self.encoder_hidden_size = encoder_hidden_size,
        self.decoder_hidden_siz =  decoder_hidden_size,
        self.encoder_num_layers = encoder_num_layers,
        self.encoder_bias = encoder_bias,
        self.decoder_num_layer = sdecoder_num_layers,
        self.decoder_bias = decoder_bias,
        self.encoder_dropout = encoder_dropout,
        self.decoder_dropout = decoder_dropout,
        self.encoder_bidirectional = encoder_bidirectional,
        self.decoder_bidirectional = decoder_bidirectional,
        self.encoder = nn.LSTM(
            self.input_size, lstm_hidden_size, batch_first=self.batch_first, num_layers=lstm_layers
        )
        self.decoder = nn.LSTM(
            self.input_size, lstm_hidden_size, batch_first=self.batch_first, num_layers=lstm_layers
        )

    def forward(self, input, h0_c0)
        h0, c0 = h0_c0
        self.encoder(
    
class EncoderRNN(nn.Module):
    def __init__(self, lstm_hidden_size, lstm_layers=2, device="cpu"):
        super(EncoderRNN, self).__init__()
        self.lstm_layers = lstm_layers
        self.device = device
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(
            34, lstm_hidden_size, batch_first=True, num_layers=lstm_layers
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
