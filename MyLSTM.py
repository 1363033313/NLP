import torch
from torch import nn
from torch.utils.data import DataLoader
from word_embedding import load_imdb


def init_params(shape):
    return torch.zeros(shape, requires_grad=True, device='cuda')


def three(num_inputs, num_hiddens):
    return torch.nn.Parameter(init_params((num_inputs, num_hiddens))), \
           torch.nn.Parameter(init_params((num_hiddens, num_hiddens))), \
           torch.nn.Parameter(init_params((num_hiddens)))


def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))


def lstm_calculate(inputs, params, state):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c] = params
    (H, C) = state
    for X in inputs:
        I = torch.sigmoid(torch.matmul(X, W_xi) + torch.matmul(H, W_hi) + b_i)
        F = torch.sigmoid(torch.matmul(X, W_xf) + torch.matmul(H, W_hf) + b_f)
        O = torch.sigmoid(torch.matmul(X, W_xo) + torch.matmul(H, W_ho) + b_o)
        C_tilda = torch.tanh(torch.matmul(X, W_xc) + torch.matmul(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
    return H


class LSTMModel(nn.Module):
    """A RNN Model implemented from scratch."""

    def __init__(self, vocab_size, num_hiddens, init_state, forward_fn=lstm_calculate,
                 embedded_size=100, *args, **kwargs):
        """Defined in :numref:`sec_rnn_scratch`"""
        super().__init__(*args, **kwargs)
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.W_xi, self.W_hi, self.b_i = three(embedded_size, num_hiddens)  # 输入门参数
        self.W_xf, self.W_hf, self.b_f = three(embedded_size, num_hiddens)  # 遗忘门参数
        self.W_xo, self.W_ho, self.b_o = three(embedded_size, num_hiddens)  # 输出门参数
        self.W_xc, self.W_hc, self.b_c = three(embedded_size, num_hiddens)  # 候选记忆元参数
        self.xavier_init()
        self.embedding = nn.Embedding.from_pretrained(self.glove.get_vecs_by_tokens(vocab.get_itos()),
                                                      padding_idx=vocab['<pad>'])
        self.init_state, self.forward_fn = init_state, forward_fn
        self.dense = nn.Linear(num_hiddens, 2)

    def __call__(self, inputs, state):
        inputs = self.embedding(inputs).transpose(0, 1)
        (H, C) = state
        for X in inputs:
            I = torch.sigmoid(torch.matmul(X, self.W_xi) + torch.matmul(H, self.W_hi) + self.b_i)
            F = torch.sigmoid(torch.matmul(X, self.W_xf) + torch.matmul(H, self.W_hf) + self.b_f)
            O = torch.sigmoid(torch.matmul(X, self.W_xo) + torch.matmul(H, self.W_ho) + self.b_o)
            C_tilda = torch.tanh(torch.matmul(X, self.W_xc) + torch.matmul(H, self.W_hc) + self.b_c)
            C = F * C + I * C_tilda
            H = O * torch.tanh(C)
        return self.dense(H)

    def xavier_init(self):
        nn.init.xavier_uniform_(self.W_xi)
        nn.init.xavier_uniform_(self.W_hi)
        nn.init.xavier_uniform_(self.W_xf)
        nn.init.xavier_uniform_(self.W_hf)
        nn.init.xavier_uniform_(self.W_xo)
        nn.init.xavier_uniform_(self.W_ho)
        nn.init.xavier_uniform_(self.W_xc)
        nn.init.xavier_uniform_(self.W_hc)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 14
train_data, test_data, vocab = load_imdb()
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size, num_hiddens = len(vocab), 256

model = LSTMModel(len(vocab), num_hiddens, init_lstm_state).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(1, NUM_EPOCHS + 1):
    print(f'Epoch {epoch}\n' + '-' * 32)
    avg_train_loss = 0
    for batch_idx, (X, y) in enumerate(train_loader):
        state = model.begin_state(batch_size=X.shape[0], device=device)
        X, y = X.to(device), y.to(device)
        pred = model(X, state)
        loss = criterion(pred, y)
        avg_train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 5 == 0:
            print(f"[{(batch_idx + 1) * BATCH_SIZE:>5}/{len(train_loader.dataset):>5}] train loss: {loss:.4f}")
    print(f"Avg train loss: {avg_train_loss / (batch_idx + 1):.4f}\n")

acc = 0
for X, y in test_loader:
    with torch.no_grad():
        state = model.begin_state(batch_size=X.shape[0], device=device)
        X, y = X.to(device), y.to(device)
        pred = model(X, state)
        acc += (pred.argmax(1) == y).sum().item()
print(f"Accuracy: {acc / len(test_loader.dataset):.4f}")
