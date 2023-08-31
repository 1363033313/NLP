### 代码应用与实践任务其一

分类任务是最简单也最易学习的应用场景，这个代码任务可以分类任务来帮助大家入门NLP这一领域。

（1） IMDB情感分析任务基本上算是一个已经被刷烂掉了的任务，不过也是很好的入门学习任务

1. 从[官网](https://ai.stanford.edu/~amaas/data/sentiment/)上下载数据集
2. 由于文本的离散特性，往往需要先利用一些库对文本进行tokenize
3. 根据tokenize之后的token得到对应的word embedding
4. 从word embedding开始接入常规的模型训练过程
5. 使用基本的textcnn、lstm实现，准确率不做要求

（2）学习最基本的预训练模型，bert和gpt**选其一**（多做加分，~~全选最好~~）

1. 掌握transformer模型原理

2. 自己动手实现bert/gpt以及第一问实现的tokenizer完成IMDB分类任务

   * 从头开始训练的bert/gpt最后准确率较差属于正常现象，不必过分纠结调参

3. 使用预训练好的bert/gpt以及它们对应的tokenizer进行finetune，完成IMDB分类任务

   * 调用的库不做要求，强烈建议使用transformers库

   * [transformers库官方文档](https://huggingface.co/docs/transformers/training) ，官方文本分类[代码参考](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/)
   * 不能直接使用已经在IMDB上finetune好的模型权重
   * 开源的gpt预训练权重用的别较多的是gpt2，跟gpt模型上没有太大区别

4. 比较bert和gpt的区别

说明：

* 我们不做准确率要求，但希望你在使用非finetune方法（textcnn/lstm/自己实现的bert/gpt）最高能达到90%或者使用finetune方法能达到92%，**前者达到92%或者后者达到95%的正确率可直接获得面试资格**（此项请于截止日期前将结果提交给相关人员核验以确定直通通道资格）
* **没做完或者准确率很低也没关系！**我们希望你每个完成的部分都有完整可运行的代码，**态度最重要**









（1） IMDB情感分析任务基本上算是一个已经被刷烂掉了的任务，不过也是很好的入门学习任务

1. 从[官网](https://ai.stanford.edu/~amaas/data/sentiment/)上下载数据集
2. 由于文本的离散特性，往往需要先利用一些库对文本进行tokenize
3. 根据tokenize之后的token得到对应的word embedding
4. 从word embedding开始接入常规的模型训练过程
5. 使用基本的textcnn、lstm实现，准确率不做要求



1234使用torchtext定义的库函数，以及预定义好的nn.Embedding层可以完成实现

### LSTM的实现

LSTM的最主要实现流程图如下所示：

![Dive Into DL](./img/LSTM.jpg)

首先是定义了遗忘门，输入门，输出门，以及候选记忆单元：
$$
\mathbf{I}_{t} =\sigma(\mathbf{X}_t\mathbf{W}_{xi}+\mathbf{H}_{t-1}\mathbf{W}_{hi}+\mathbf{b}_i), \\
\mathbf{F}_{t} =\sigma(\mathbf{X}_t\mathbf{W}_{xf}+\mathbf{H}_{t-1}\mathbf{W}_{hf}+\mathbf{b}_f), \\
\mathbf{O}_{t} =\sigma(\mathbf{X}_t\mathbf{W}_{xo}+\mathbf{H}_{t-1}\mathbf{W}_{ho}+\mathbf{b}_o),\\
\tilde{\mathbf{C}}_t=\tanh(\mathbf{X}_t\mathbf{W}_{xc}+\mathbf{H}_{t-1}\mathbf{W}_{hc}+\mathbf{b}_c),
$$
其中$\sigma(x)$为$sigmoid$激活函数 ，这些计算实际上和RNN中隐状态的计算比较类似

下面我们定义记忆元
$$
\mathbf{C}_t=\mathbf{F}_t\odot\mathbf{C}_{t-1}+\mathbf{I}_t\odot\tilde{\mathbf{C}}_t
$$
$\mathbf{F}_t\odot\mathbf{C}_{t-1}$实际上表示了遗忘多少先前的记忆元$\mathbf{C}_{t-1}$，而$\mathbf{I}_t\odot\tilde{\mathbf{C}}_t$ 则代表了当前候选记忆元使用的程度

$LSTM$中隐状态的定义如下：
$$
\mathbf{H}_t=\mathbf{O}_t\odot\tanh(\mathbf{C}_t)
$$
这样的定义可以保证每个元素均在$[-1,1]$之间，防止梯度爆炸

下面我们从零实现$LSTM$, 参考了李沐老师的[14. 自然语言处理：预训练 — 动手学深度学习 2.0.0 documentation (d2l.ai)](http://zh-v2.d2l.ai/chapter_natural-language-processing-pretraining/index.html)()

首先我们定义初始参数的函数：

```python
def init_params(shape):
    return torch.zeros(shape, requires_grad=True, device='cuda')


def three(num_inputs, num_hiddens):
    return torch.nn.Parameter(init_params((num_inputs, num_hiddens))), \
           torch.nn.Parameter(init_params((num_hiddens, num_hiddens))), \
           torch.nn.Parameter(init_params((num_hiddens)))
```

这个函数用于初始记忆元和隐状态

```python
def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))
```

下面定义了$LSTM$层的计算步骤，基本上就是按照$LSTM$定义的方法实现的

```python
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
```

有了以上代码，可以实现我们的$LSTM$模型了

```python
lass LSTMModel(nn.Module):
    """A RNN Model implemented from scratch."""

    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn=lstm_calculate,
                 embedded_size=100, *args, **kwargs):
        """Defined in :numref:`sec_rnn_scratch`"""
        super().__init__(*args, **kwargs)
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.W_xi, self.W_hi, self.b_i = three(embedded_size, num_hiddens)  # 输入门参数
        self.W_xf, self.W_hf, self.b_f = three(embedded_size, num_hiddens)  # 遗忘门参数
        self.W_xo, self.W_ho, self.b_o = three(embedded_size, num_hiddens)  # 输出门参数
        self.W_xc, self.W_hc, self.b_c = three(embedded_size, num_hiddens)  # 候选记忆元参数
        self.xavier_init()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embedded_size)
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
```

下面是训练及测试代码：

```python
batch_size = 512
lr = 0.0005
epochs = 40
train_data, test_data, vocab = load_imdb()
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size, num_hiddens = len(vocab), 256

model = LSTMModel(len(vocab), num_hiddens, init_lstm_state).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=1000, num_training_steps=10000)

for epoch in range(epochs):
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
    print(f"Epoch {epoch + 1} Avg train loss: {avg_train_loss / (batch_idx + 1):.4f}")
    acc = 0
    for X, y in test_loader:
        with torch.no_grad():
            state = model.begin_state(batch_size=X.shape[0], device=device)
            X, y = X.to(device), y.to(device)
            pred = model(X, state)
            acc += (pred.argmax(1) == y).sum().item()
    print(f"Epoch {epoch + 1} Test Accuracy: {acc / len(test_loader.dataset):.4f}\n")

```

训练的结果如下：

![](D:\dlstudy\NLP\img\MyLSTM.png)

真的很怪，训练了很多次损失就是不下降，一直在0.69-0.67之间波动，但上图这次又不知道为什么正常了起来，可能这就是炼丹吧（）

### $TextCNN$的实现

网络结构大致如下：

![](./img/textcnn.jpg)

第一层：

输入层，每个词向量可以是预先在其他语料库中训练好的，也可以作为未知的参数由网络训练得到。预先训练的词嵌入可以利用其他语料库得到更多的先验知识，而由当前网络训练的词向量能够更好地抓住与当前任务相关联的特征。因此，图中的输入层实际采用了双通道的形式，即有两个 $N\times k$ 的输入矩阵，其中一个用预训练好的词嵌入表达，并且在训练过程中不再发生变化；另外一个也由同样的方式初始化，但是会作为参数，随着网络的训练过程发生改变

实现如下：

```python
self.glove = GloVe(name="6B", dim=100)
        self.embedding_constant = nn.Embedding.from_pretrained(self.glove.get_vecs_by_tokens(vocab.get_itos()),
                                                               padding_idx=vocab['<pad>'], freeze=True)
        self.embedding_changing = nn.Embedding(len(vocab),embedding_dim=100, padding_idx=vocab['<pad>'])
```

其中第一个embedding层使用预训练的100维Glove词向量，第二个embedding层使用自己初始化的embedding层，让其自己学习单词的向量表示

第二层：

卷积层，把embedded_size当作通道数，对每个通道进行一维卷积，通道求和得到输出，我们可以得到多个输出通道以提升模型的复杂度，让模型提取不同的语义信息

这里偷个懒直接使用二维卷积了，实际上效果差不太多

```python
self.conv_constant = nn.ModuleList()
        self.conv_changing = nn.ModuleList()
        for out_channels, kernel_size in zip(num_channels, kernel_sizes):
            self.conv_constant.append(
                nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(kernel_size, embed_size)))
            self.conv_changing.append(
                nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(kernel_size, embed_size)))
```

第三层：

池化连接层，将上一步的输出用最大池化，再进行连接，加上dense层输出二分类的概率即可、

```python
self.pool = nn.AdaptiveMaxPool1d(1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(sum(num_channels) * 2, 2)
```

![](D:\dlstudy\NLP\img\TextCNN.png)



### $BERT$的实现

BERT实际上就是更大的transformer解码器，在解码器的最后加上dense层就可得到输出了

实现主要参考了transformers库的源代码

Config主要保存了各种配置信息，方便更新调整

```python
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
          hidden_size: Size0 of the encoder layers and the pooler layer.
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
```

#### Input Embedding

Input Embedding=PositionalEmbedding(位置编码)+TokenEmbedding（词向量）+SegmentEmbedding（区分句子）



```python
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
```

#### BertAttention 实现

核心就是在 Transformer 中所提出来的 self-attention 机制

```python
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
```

#### FFN的实现

实际上就是一个MLP

```python
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
```

#### 组装Encoder

把以上组件组装起来就可以了

```python
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
```

#### 池化输出

对最后的隐藏状态进行平均池化，加上dense层得到输出

```python
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
        return pooled_oeutput  # [batch_size, 2]
```

#### BERT模型实现

组装Embedding+Encoder+Pool

```python
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
```

#### 训练代码

训练集有所不同，句首加上了[CLS]句尾加上了[SEP]，与论文相同

scheduler不加不行，不加的话损失一直降不下去（）

```python
batch_size = 64
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
    print(f'Epoch {epoch + 1} Avg train loss: {avg_train_loss / (batch_idx + 1):.4f}')
    acc = 0
    for X, y in test_loader:
        with torch.no_grad():
            X, y = X.to(devices), y.to(devices)
            pred, _ = BERTNet(X)
            acc += (pred.argmax(1) == y).sum().item()

    print(f"Epoch {epoch + 1} Test Accuracy: {acc / len(test_loader.dataset):.4f}\n")
```

训练的结果大致如下，确实效果比较一般（）：

![](D:\dlstudy\NLP\img\bert.png)

### Fine-tune

可以调库就比较轻松了，调用预选连模型和分词器训练就可以了，修改了一下数据集的载入

```python
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
    print(f'Avg train loss: {avg_train_loss / (batch_idx + 1):.4f}\n')
    model.eval()
    acc = 0
    for X, mask, y in test_loader:
        with torch.no_grad():
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            pred = model(X, token_type_ids=None, attention_mask=mask, labels=y).logits
            acc += (pred.argmax(1) == y).sum().item()
    print(f"Accuracy: {acc / len(test_loader.dataset):.4f}\n")
```

![	](D:\dlstudy\NLP\img\Fine_tune.png)

fine_tune效果也比较一般（）

### BERT和GPT的区别

1. BERT是Transformer编码器，GPT是Transformer解码器
2. 预训练差距比较大，BERT是做完形填空，GPT在做预测未来（标准的语言模型），后者显然要更难一些，这可能也是GPT的效果要比BERT差一些的原因之一吧



