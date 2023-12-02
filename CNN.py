import torch 
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        embedding_dim: int, 
        num_filters: int,
        filter_sizes: list,
        output_dim: int,
        dropout_probability: float,
        padding_idx: int,
        name: str
    ):
        super(Classifier, self).__init__()
        self.name = name

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=padding_idx)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(filter_size, embedding_dim))
            for filter_size in filter_sizes
        ])

        self.fully_connected = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, sequences):

        # (batch_size, sequence_dim) ->
        # we use unsqueeze to add a dimension so that the Conv2d layers can process
        embedded = self.dropout(self.embedding(sequences.unsqueeze(1)))

        # (batch_size, 1, sequence_dim, embedding_dim) ->
        conv_outputs = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conv_outputs: list of (batch_size, num_filters, convolved_length) ->
        pooled_outputs = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conv_outputs]

        # pooled_outputs: list of (batch_size, num_filters) ->
        cat = self.dropout(torch.cat(pooled_outputs, dim=1))

        # (batch_size, len(filter_sizes) * num_filters) ->
        predictions = self.fully_connected(cat)

        # (batch_size, 1) ->
        return predictions