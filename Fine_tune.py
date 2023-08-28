import os

import torch
import torch.nn as nn
from torch.optim import AdamW

from torch.utils.data import TensorDataset, Dataset, DataLoader
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup, BertTokenizer

from word_embedding import set_seed


def read_imdb(path='./aclImdb', is_train=True):
    reviews, labels = [], []
    for label in ['pos', 'neg']:
        folder_name = os.path.join(path, 'train' if is_train else 'test', label)
        for filename in os.listdir(folder_name):
            with open(os.path.join(folder_name, filename), mode='r', encoding='utf-8') as f:
                reviews.append(f.read())
                labels.append(1 if label == 'pos' else 0)
    return reviews, labels


def load_imdb():
    train, train_labels = read_imdb(is_train=True)
    test, test_labels = read_imdb(is_train=False)
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    train_data = tokenizer(train, padding=True, truncation=True, max_length=512, return_tensors="pt")
    test_data = tokenizer(test, padding=True, truncation=True, max_length=512, return_tensors="pt")
    train_dataset = TensorDataset(train_data['input_ids'], train_data['attention_mask'], torch.tensor(train_labels))
    test_dataset = TensorDataset(test_data['input_ids'], test_data['attention_mask'], torch.tensor(test_labels))
    return train_dataset, test_dataset


set_seed(42)

train_data, test_data = load_imdb()
device = torch.device('cuda')
batch_size = 32
learning_rate = 5e-5
epochs = 5

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2,
                                                      output_attentions=False, output_hidden_states=False).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=epochs * len(train_loader))
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    print(f'Epoch {epoch + 1}:')
    avg_train_loss = 0
    for batch_idx, (X, mask, y) in enumerate(train_loader):
        X, mask, y = X.to(device), mask.to(device), y.to(device)
        pred = model(X, token_type_ids=None, attention_mask=mask, labels=y).logits
        loss = criterion(pred, y)
        avg_train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        print(f'{loss.item():.4f}')
    print(f'Avg train loss: {avg_train_loss / (batch_idx + 1):.4f}\n')
    model.eval()
    acc = 0
    for X, mask, y in test_loader:
        with torch.no_grad():
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            pred = model(X, token_type_ids=None, attention_mask=mask, labels=y).logits
            acc += (pred.argmax(1) == y).sum().item()
    print(f"Accuracy: {acc / len(test_loader.dataset):.4f}")
