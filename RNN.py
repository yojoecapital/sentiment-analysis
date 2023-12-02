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
        rnn_layers: int, 
        bidirectional: bool,
        dropout_probability: float,
        padding_idx: int
    ):
        super(Classifier, self).__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=padding_idx)

        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            num_layers=rnn_layers,
            bidirectional=bidirectional,
            dropout=dropout_probability
        )

        self.dropout = nn.Dropout(dropout_probability)
        self.fully_connected = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, sequences, true_lengths):

        # (sequence_dim, batch_size) ->
        embedded = self.dropout(self.embedding(sequences))

        # (sequence_dim, batch_size, embedding_dim) ->
        output, hidden = self.rnn(embedded)

        # hidden: (D * rnn_layers * hidden_dim) ->
        # where D = 2 if bidirectional else 1
        # concatenate hidden states if bidirectional
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        
        # (batch_size, D * hidden_dim) ->
        predictions = self.fully_connected(hidden)
        
        # (batch_size, 1)
        return predictions
