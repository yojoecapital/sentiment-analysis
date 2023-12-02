import torch 
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        embedding_dim: int, 
        hidden_dim: int, 
        output_dim: int,
        lstm_layers: int, 
        bidirectional: bool, 
        dropout_probability: float, 
        padding_idx: int,
        name: str
    ):
        super(Classifier, self).__init__()
        self.name = name

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=padding_idx)

        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            dropout=dropout_probability
        )

        self.dropout = nn.Dropout(dropout_probability)
        
        self.fully_connected = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, sequences, true_lengths):

        # (sequence_dim, batch_size) ->
        embedded = self.dropout(self.embedding(sequences))

        # (sequence_dim, batch_size, embedding_dim) ->

        # pack_padded_sequence packes the embedding by only using the true sequence lengths
        # i.e. getting rid of the padded tokens
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, true_lengths.to('cpu'))
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # unpack sequence to a tensor
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # hidden: (D * lstm_layers, hidden_him) ->
        # where D = 2 if bidirectional else 1
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # (batch_size, D * hidden_dim) ->
        predictions = self.fully_connected(hidden)

        # (batch_size, 1)
        return predictions
