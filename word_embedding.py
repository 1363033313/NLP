import os
import random
import numpy as np
import torch
import torchtext
from torchtext.vocab import build_vocab_from_iterator
from torchtext import transforms as T
from torch.utils.data import TensorDataset


def read_imdb(path='./aclImdb', is_train=True):
    reviews, labels = [], []
    tokenizer = torchtext.data.get_tokenizer("basic_english")
    for label in ['pos', 'neg']:
        folder_name = os.path.join(path, 'train' if is_train else 'test', label)
        for filename in os.listdir(folder_name):
            with open(os.path.join(folder_name, filename), mode='r', encoding='utf-8') as f:
                reviews.append(tokenizer(f.read()))
                labels.append(1 if label == 'pos' else 0)
    return reviews, labels


def build_dataset(reviews, labels, vocab, max_len=512, bert_preprocess=False):
    if bert_preprocess:
        text_transform = T.Sequential(
            T.VocabTransform(vocab=vocab),
            T.Truncate(max_seq_len=max_len - 2),  # 之所以减2是因为接下来要添加两个特殊词元
            T.AddToken(token=vocab['<cls>'], begin=True),
            T.AddToken(token=vocab['<sep>'], begin=False),
            T.ToTensor(padding_value=vocab['<pad>']),
            T.PadTransform(max_length=max_len, pad_value=vocab['<pad>']),
        )
    else:
        text_transform = T.Sequential(
            T.VocabTransform(vocab=vocab),
            T.Truncate(max_seq_len=max_len),
            T.ToTensor(padding_value=vocab['<pad>']),
            T.PadTransform(max_length=max_len, pad_value=vocab['<pad>']),
        )
    dataset = TensorDataset(text_transform(reviews), torch.tensor(labels))
    return dataset


def load_imdb(bert_preprocess=False):
    reviews_train, labels_train = read_imdb(is_train=True)
    reviews_test, labels_test = read_imdb(is_train=False)
    vocab = build_vocab_from_iterator(reviews_train, min_freq=3, specials=['<pad>', '<unk>', '<cls>', '<sep>'])
    vocab.set_default_index(vocab['<unk>'])
    train_data = build_dataset(reviews_train, labels_train, vocab, bert_preprocess=bert_preprocess)
    test_data = build_dataset(reviews_test, labels_test, vocab, bert_preprocess=bert_preprocess)
    return train_data, test_data, vocab


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# train_data, test_data, vocab = load_imdb()
