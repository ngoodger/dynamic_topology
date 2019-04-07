#import seq2seq
import pytest
from torch.utils.data import Dataset, DataLoader
import random
import torch
from torch.nn.utils.rnn import pack_padded_sequence

def pack_unsorted_padded_sequence(input, lengths, batch_first=False):
    seq_sorted_index = torch.argsort(lengths, dim=0, descending=True)
    # Reindex by new order.
    sorted_lengths = lengths[seq_sorted_index]
    sorted_input = input[seq_sorted_index, :, :]
    torch.arange(len(lengths))
    # Try to pack input into sequence.
    pack = torch.nn.utils.rnn.pack_padded_sequence(sorted_input, sorted_lengths, batch_first=batch_first)
    return pack
    

class Seq2SeqToyDataset(Dataset):
    """
    Toy example where seq2seq model is effective.
    Number translation from a langauge with 8 numbers to language with 4.
    First 2 values are occupied by Start Of Sequence (SOS) and End Of Sequence (EOS) tokens .

    The mapping is as follows.
    2 = 4
    3 = 3, 3
    4 = 5, 2
    5 = 2
    6 = 5
    7 = 3, 5
    8 = 4, 2
    9 = 5, 4

    However to ensure the model the model is capable the output is also sorted
    in the same way word cord can vary in language.
    i.e.
    
    input sequence   ==>   Direct translation   ==>  Reorder translation 
    2 3 4            ==>   4 33 52              ==>  2 3 3 4 5
    """ 
    # Start and EOS token values
    INPUT_WORD_COUNT = 8
    TARGET_WORD_COUNT = 4
    SOS = 0 
    EOS = 1 
    TOKEN_COUNT = 2
    MAX_INPUT_SEQ_LEN = 8
    MIN_INPUT_SEQ_LEN = 1
    MIN_INPUT_TO_TARGET_TOKEN_EXPANSION = 1
    MAX_INPUT_TO_TARGET_TOKEN_EXPANSION = 2
    MIN_TARGET_SEQ_LEN = MIN_INPUT_TO_TARGET_TOKEN_EXPANSION * MIN_INPUT_SEQ_LEN
    MAX_TARGET_SEQ_LEN = MAX_INPUT_TO_TARGET_TOKEN_EXPANSION * MAX_INPUT_SEQ_LEN
    EXAMPLE_COUNT = 1024

    def __init__(self):
        random.seed(0)
        super(Seq2SeqToyDataset, self).__init__()

    def __len__(self):
        return self.EXAMPLE_COUNT

    def __getitem__(self, idx):
        # Mapping as above.
        input_target_map = {2: [4],
                            3 : [3, 3],
                            4: [5, 2],
                            5: [2],
                            6: [5],
                            7: [3, 5],
                            8: [4, 2],
                            9: [5, 4]}
        input_seq_len_no_token = random.randint(self.MIN_INPUT_SEQ_LEN, self.MAX_INPUT_SEQ_LEN)
        inputs, direct_targets = [], []
        for input_seq_idx in range(input_seq_len_no_token):
            input_value = random.randint(self.TOKEN_COUNT, self.INPUT_WORD_COUNT + self.TOKEN_COUNT - 1)
            inputs.append(input_value)
            direct_targets += input_target_map[input_value]
        # Sort target by size. 
        targets = sorted(direct_targets)

        # Add tokens.
        inputs.insert(0, self.SOS)
        inputs.append(self.EOS)
        targets.insert(0, self.SOS)
        targets.append(self.EOS)

        # Determine sequence lengths including tokens.
        input_seq_len = len(inputs)
        target_seq_len = len(targets)

        # Zero pad sequences.
        inputs += [0] * ((self.MAX_INPUT_SEQ_LEN + self.TOKEN_COUNT) - len(inputs))
        targets += [0] * ((self.MAX_TARGET_SEQ_LEN + self.TOKEN_COUNT) - len(targets))

        # Convert inputs and targets to tensors.  Add data vector dimension so number of dimensions is consistent for higher order inputs and targets.
        input_tensor = torch.Tensor(inputs).unsqueeze(1)
        target_tensor = torch.Tensor(targets).unsqueeze(1)

        data = {"input_seq_len": input_seq_len, "input_tensor": input_tensor,
                "target_seq_len": target_seq_len, "target_tensor": target_tensor} 
        return data

def test_toy_example():
    BATCH_SIZE = 128 
    dataset = Seq2SeqToyDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)
    data = next(iter(dataloader))
    input_seq_len = data["input_seq_len"]
    input_tensor = data["input_tensor"]
    target_seq_len = data["target_seq_len"]
    target_tensor = data["target_tensor"]
    assert (len(input_tensor.shape) == 3), "Dimensions should be batch index, sequence index, input vector index"
    assert (len(target_tensor.shape) == 3), "Dimensions should be batch index, sequence index, target vector index" 
    assert (len(input_seq_len.shape) == 1), "Dimensions should be batch index"
    assert (len(target_seq_len.shape) == 1), "Dimensions should be batch index" 

    assert (input_tensor.shape[1] == Seq2SeqToyDataset.MAX_INPUT_SEQ_LEN + Seq2SeqToyDataset.TOKEN_COUNT)
    assert (target_tensor.shape[1] == Seq2SeqToyDataset.MAX_TARGET_SEQ_LEN + Seq2SeqToyDataset.TOKEN_COUNT)

    # Inputs and targets have size of 1.  (Loss function handles one-hot encoding)
    assert (input_tensor.shape[2] == 1) 
    assert (target_tensor.shape[2] == 1) 

    # Check input values are valid and zero padding is correct.
    for batch_idx in range(BATCH_SIZE):
        for seq_idx in range(Seq2SeqToyDataset.MAX_INPUT_SEQ_LEN):
            if seq_idx < input_seq_len[batch_idx]:
                assert (input_tensor[batch_idx, seq_idx, 0] in range(Seq2SeqToyDataset.INPUT_WORD_COUNT + Seq2SeqToyDataset.TOKEN_COUNT)), "Input value is not valid"
            else:
                assert (input_tensor[batch_idx, seq_idx, 0] == 0), "Value should be zero padding."

    # Check target values are valid and zero padding is correct.
    for batch_idx in range(BATCH_SIZE):
        for seq_idx in range(Seq2SeqToyDataset.MAX_TARGET_SEQ_LEN):
            if seq_idx < target_seq_len[batch_idx]:
                assert (target_tensor[batch_idx, seq_idx, 0] in range(Seq2SeqToyDataset.TARGET_WORD_COUNT + Seq2SeqToyDataset.TOKEN_COUNT)), "Target value is not valid"
            else:
                assert (target_tensor[batch_idx, seq_idx, 0] == 0), "Value should be zero padding."

    # Check each possible input value appears in input (Batch size must be sufficiently large for this to occur).
    for input_value in range(Seq2SeqToyDataset.INPUT_WORD_COUNT + Seq2SeqToyDataset.TOKEN_COUNT):
            assert ((input_value == input_tensor).any()), "All possible input tokens must appear in input tensor."

    # Check each possible target value appears in input (Batch size must be sufficiently large for this to occur).
    for target_value in range(Seq2SeqToyDataset.TARGET_WORD_COUNT + Seq2SeqToyDataset.TOKEN_COUNT):
            assert ((target_value == target_tensor).any()), "All possible target tokens must appear in target tensor."

    # Check each possible input sequence length value appears in input sequence.  (Batch size must be sufficiently large for this to occur).
    for input_seq_len_value in range(Seq2SeqToyDataset.MIN_INPUT_SEQ_LEN + Seq2SeqToyDataset.TOKEN_COUNT, Seq2SeqToyDataset.MAX_INPUT_SEQ_LEN + Seq2SeqToyDataset.TOKEN_COUNT):
            assert ((input_seq_len_value == input_seq_len).any()), "All possible input sequence lengths must appear in input_seq_len"

    # Check each possible target sequence length value appears in target sequence.  (Batch size must be sufficiently large for this to occur).
    for target_seq_len_value in range(Seq2SeqToyDataset.MIN_TARGET_SEQ_LEN + Seq2SeqToyDataset.TOKEN_COUNT, Seq2SeqToyDataset.MAX_TARGET_SEQ_LEN + Seq2SeqToyDataset.TOKEN_COUNT):
            assert ((target_seq_len_value == target_seq_len).any()), "All possible target sequence lengths must appear in target_seq_len"

    input_pack = pack_unsorted_padded_sequence(input_tensor, input_seq_len, batch_first=True)
    target_pack = pack_unsorted_padded_sequence(input_tensor, input_seq_len, batch_first=True)
    print(input_pack)
    print(target_pack)

def test_seq2seq():
    """
    Test seq2seq model is able to learn simple
    """
    pass
    """
    seq2seq_dut = seq2seq.Seq2Seq(input_size=2,
                                  encoder_hidden_size=8,
                                  decoder_hidden_size=16,
                                  batch_first=True,
                                  encoder_num_layers=2,
                                  decoder_num_layers=2,
                                  )
    
    """
