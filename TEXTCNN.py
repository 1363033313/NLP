import random
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from torchtext.vocab import GloVe

from word_embedding import load_imdb, set_seed


def _init_weights(m):
    if type(m) in (nn.Linear, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


class TextCNN(nn.Module):
    def __init__(self, vocab, embed_size=100, kernel_sizes=None, num_channels=None):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 4, 5]
        if num_channels is None:
            num_channels = [100, 100, 100]
        self.embedding_constant = nn.Embedding(len(vocab), 100, padding_idx=vocab['<pad>'])
        self.glove = GloVe(name="6B", dim=100)
        self.embedding_changing = nn.Embedding.from_pretrained(self.glove.get_vecs_by_tokens(vocab.get_itos()),
                                                               padding_idx=vocab['<pad>'],
                                                               freeze=True)
        self.conv_constant = nn.ModuleList()
        self.conv_changing = nn.ModuleList()
        for out_channels, kernel_size in zip(num_channels, kernel_sizes):
            self.conv_constant.append(
                nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(kernel_size, embed_size)))
            self.conv_changing.append(
                nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(kernel_size, embed_size)))

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(sum(num_channels) * 2, 2)
        self.apply(_init_weights)

    def forward(self, x):
        x_unfrozen = self.embedding_changing(x).unsqueeze(1)  # (batch_size, seq_len, embed_size)
        x_frozen = self.embedding_constant(x).unsqueeze(1)  # (batch_size, embed_size, seq_len)
        pool_con = [self.pool(self.relu(conv(x_unfrozen).squeeze())).squeeze()
                    for conv in self.conv_changing]  # (batch_size, 100)
        pool_cha = [self.pool(self.relu(conv(x_frozen).squeeze())).squeeze()
                    for conv in self.conv_constant]  # shape of each element: (batch_size, 100)
        feature = torch.cat(pool_con + pool_cha, dim=-1)
        output = self.fc(self.dropout(feature))
        return output


set_seed(42)

batch_size = 256
num_epochs = 40

train_data, test_data, vocab = load_imdb()
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TextCNN(vocab).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0009, weight_decay=5e-4)

for epoch in range(num_epochs):
    avg_train_loss = 0
    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        avg_train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1} Avg train loss: {avg_train_loss / (batch_idx + 1):.4f}")
    acc = 0
    for X, y in test_loader:
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            pred = model(X)
            acc += (pred.argmax(1) == y).sum().item()
    print(f"Epoch {epoch + 1} Test Accuracy: {acc / len(test_loader.dataset):.4f}\n")
