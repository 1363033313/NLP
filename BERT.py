import torch.nn as nn
import torch
from torch.nn.init import normal_
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe

import json
import copy
import six
import logging

from transformers import get_scheduler

from word_embedding import load_imdb, load_imdb_bert


class BertConfig(object):
    """Configuration for `BertModel`."""

    def __init__(self,
                 vocab,
                 hidden_size=100,
                 num_hidden_layers=6,
                 num_attention_heads=4,
                 intermediate_size=1024,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs BertConfig.
        Args:
          vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
          hidden_size: Size of the encoder layers and the pooler layer.
          num_hidden_layers: Number of hidden layers in the Transformer encoder.
          num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder.
          intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
          hidden_act: The non-linear activation function (function or string) in the
            encoder and pooler.
          hidden_dropout_prob: The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
          attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
          max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
          type_vocab_size: The vocabulary size of the `token_type_ids` passed into
            `BertModel`.
          initializer_range: The stdev of the truncated_normal_initializer for
            initializing all weight matrices.
        """
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class PositionalEmbedding(nn.Module):

    def __init__(self, hidden_size, max_position_embeddings=512, initializer_range=0.02):
        super(PositionalEmbedding, self).__init__()
        # BERT预训练模型的长度为512
        self.embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self._reset_parameters(initializer_range)

    def forward(self, position_ids):  # [1,position_ids_len]
        return self.embedding(position_ids).transpose(0, 1)  # [position_ids_len, 1, hidden_size]

    def _reset_parameters(self, initializer_range):
        for p in self.parameters():
            if p.dim() > 1:
                normal_(p, mean=0.0, std=initializer_range)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab, hidden_size):
        super(TokenEmbedding, self).__init__()
        self.glove = GloVe(name="6B", dim=100)
        self.embedding = nn.Embedding.from_pretrained(self.glove.get_vecs_by_tokens(vocab.get_itos()),
                                                      padding_idx=vocab['<pad>'])

    def forward(self, input_ids):  # [batch_size, input_ids_len]
        return self.embedding(input_ids).transpose(0, 1)  # [input_ids_len, batch_size, hidden_size]


class SegmentEmbedding(nn.Module):
    def __init__(self, type_vocab_size, hidden_size, initializer_range=0.02):
        super(SegmentEmbedding, self).__init__()
        self.embedding = nn.Embedding(type_vocab_size, hidden_size)
        self._reset_parameters(initializer_range)

    def forward(self, token_type_ids):  # [batch_size, token_type_ids_len ]
        return self.embedding(token_type_ids).transpose(0, 1)  # [token_type_ids_len, batch_size, hidden_size]

    def _reset_parameters(self, initializer_range):
        for p in self.parameters():
            if p.dim() > 1:
                normal_(p, mean=0.0, std=initializer_range)


class BertEmbeddings(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : normal embedding matrix
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = TokenEmbedding(vocab=config.vocab,
                                              hidden_size=config.hidden_size)
        # return shape [src_len,batch_size,hidden_size]

        self.position_embeddings = PositionalEmbedding(max_position_embeddings=config.max_position_embeddings,
                                                       hidden_size=config.hidden_size,
                                                       initializer_range=config.initializer_range)
        # return shape [src_len,1,hidden_size]

        self.token_type_embeddings = SegmentEmbedding(type_vocab_size=config.type_vocab_size,
                                                      hidden_size=config.hidden_size,
                                                      initializer_range=config.initializer_range)
        # return shape  [src_len,batch_size,hidden_size]

        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids",
                             torch.arange(config.max_position_embeddings).expand((1, -1)))
        # shape: [1, max_position_embeddings]

    def forward(self, input_ids=None, position_ids=None, token_type_ids=None):
        """
        :param input_ids:   [batch_size, src_len]
        :param position_ids:  shape: [1,src_len]
        :param token_type_ids:  shape:[src_len,batch_size]
        :return: [src_len, batch_size, hidden_size]
        """
        src_len = input_ids.size(1)
        token_embedding = self.word_embeddings(input_ids)
        # shape:[src_len,batch_size,hidden_size]

        if position_ids is None:
            position_ids = self.position_ids[:, :src_len]  # [1,src_len]
        positional_embedding = self.position_embeddings(position_ids)
        # [src_len, 1, hidden_size]

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids,
                                              device=self.position_ids.device)  # [src_len, batch_size]
        segment_embedding = self.token_type_embeddings(token_type_ids)
        # [src_len,batch_size,hidden_size]

        embeddings = token_embedding + positional_embedding + segment_embedding
        # [src_len,batch_size,hidden_size] + [src_len,1,hidden_size] + [src_len,batch_size,hidden_size]
        embeddings = self.LayerNorm(embeddings)  # [src_len, batch_size, hidden_size]
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):


    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=config.hidden_size,
                                                          num_heads=config.num_attention_heads,
                                                          dropout=config.attention_probs_dropout_prob)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):

        return self.multi_head_attention(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """
        :param hidden_states: [src_len, batch_size, hidden_size]
        :param input_tensor: [src_len, batch_size, hidden_size]
        :return: [src_len, batch_size, hidden_size]
        """
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def get_activation(activation_string):
    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return nn.ReLU()
    elif act == "gelu":
        return nn.GELU()
    elif act == "tanh":
        return nn.Tanh()
    else:
        raise ValueError("Unsupported activation: %s" % act)


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self,
                hidden_states,
                attention_mask=None):
        """

        :param hidden_states: [src_len, batch_size, hidden_size]
        :param attention_mask: [batch_size, src_len]
        :return: [src_len, batch_size, hidden_size]
        """
        self_outputs = self.self(hidden_states,
                                 hidden_states,
                                 hidden_states,
                                 attn_mask=None,
                                 key_padding_mask=attention_mask)
        # self_outputs[0] shape: [src_len, batch_size, hidden_size]
        attention_output = self.output(self_outputs[0], hidden_states)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        """

        :param hidden_states: [src_len, batch_size, hidden_size]
        :return: [src_len, batch_size, intermediate_size]
        """
        hidden_states = self.dense(hidden_states)  # [src_len, batch_size, intermediate_size]
        if self.intermediate_act_fn is None:
            hidden_states = hidden_states
        else:
            hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """

        :param hidden_states: [src_len, batch_size, intermediate_size]
        :param input_tensor: [src_len, batch_size, hidden_size]
        :return: [src_len, batch_size, hidden_size]
        """
        hidden_states = self.dense(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert_attention = BertAttention(config)
        self.bert_intermediate = BertIntermediate(config)
        self.bert_output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        """

        :param hidden_states: [src_len, batch_size, hidden_size]
        :param attention_mask: [batch_size, src_len] mask掉padding部分的内容
        :return: [src_len, batch_size, hidden_size]
        """
        attention_output = self.bert_attention(hidden_states, attention_mask)
        # [src_len, batch_size, hidden_size]
        intermediate_output = self.bert_intermediate(attention_output)
        # [src_len, batch_size, intermediate_size]
        layer_output = self.bert_output(intermediate_output, attention_output)
        # [src_len, batch_size, hidden_size]
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask=None):
        """

        :param hidden_states: [src_len, batch_size, hidden_size]
        :param attention_mask: [batch_size, src_len]
        :return:
        """
        all_encoder_layers = []
        layer_output = hidden_states
        for i, layer_module in enumerate(self.bert_layers):
            layer_output = layer_module(layer_output,
                                        attention_mask)
            #  [src_len, batch_size, hidden_size]
            all_encoder_layers.append(layer_output)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, 2)
        self.activation = nn.Tanh()
        self.config = config

    def forward(self, hidden_states):
        """

        :param hidden_states:  [src_len, batch_size, hidden_size]
        :return: [batch_size, hidden_size]
        """
        token_tensor = torch.mean(hidden_states, dim=0)
        pooled_output = self.dense(token_tensor)  # [batch_size, hidden_size]
        return pooled_output  # [batch_size, 2]


class BertModel(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.bert_embeddings = BertEmbeddings(config)
        self.bert_encoder = BertEncoder(config)
        self.bert_pooler = BertPooler(config)
        self.config = config
        self._reset_parameters()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None):
        """
        :param input_ids:  [src_len, batch_size]
        :param attention_mask: [batch_size, src_len]
        :param token_type_ids: [src_len, batch_size]
        :param position_ids: [1,src_len]
        :return:
        """
        embedding_output = self.bert_embeddings(input_ids=input_ids,
                                                position_ids=position_ids,
                                                token_type_ids=token_type_ids)
        # embedding_output: [src_len, batch_size, hidden_size]
        all_encoder_outputs = self.bert_encoder(embedding_output,
                                                attention_mask=attention_mask)
        sequence_output = all_encoder_outputs[-1]  # 取最后一层
        # sequence_output: [src_len, batch_size, hidden_size]
        pooled_output = self.bert_pooler(sequence_output)
        # pooled_output: [batch_size, hidden_size]
        return pooled_output, all_encoder_outputs

    def _reset_parameters(self):

        for p in self.parameters():
            if p.dim() > 1:
                normal_(p, mean=0.0, std=self.config.initializer_range)


batch_size = 32
lr = 1e-4
epochs = 40

train_data, test_data, vocab = load_imdb_bert()
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

devices = 'cuda'
config = BertConfig(vocab)
BERTNet = BertModel(config).to(devices)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(BERTNet.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-9, weight_decay=0.01)
scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=1000, num_training_steps=10000)

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}:' )
    avg_train_loss = 0
    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(devices), y.to(devices)
        pred, _ = BERTNet(X)
        loss = criterion(pred, y)
        avg_train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    print(f'Epoch {epoch + 1} Avg train loss: {avg_train_loss / (batch_idx + 1):.4f}\n')
    acc = 0
    for X, y in test_loader:
        with torch.no_grad():
            X, y = X.to(devices), y.to(devices)
            pred, _ = BERTNet(X)
            acc += (pred.argmax(1) == y).sum().item()

    print(f"Epoch {epoch + 1} Test Accuracy: {acc / len(test_loader.dataset):.4f}")

