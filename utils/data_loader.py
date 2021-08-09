from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets.translation import Multi30k


class DataLoader:
    source: Field = None
    target: Field = None

    def __init__(self, tokenizer_en, tokenizer_de, init_token, eos_token):
        self.tokenizer_en = tokenizer_en
        self.tokenizer_de = tokenizer_de
        self.init_token   = init_token
        self.eos_token    = eos_token

    def make_dataset(self):
        self.source = Field(tokenize=self.tokenizer_de, init_token=self.init_token, eos_token=self.eos_token,
                            lower=True, batch_first=True)
        self.target = Field(tokenize=self.tokenizer_en, init_token=self.init_token, eos_token=self.eos_token,
                            lower=True, batch_first=True)

        train_dataset, valid_dataset, test_dataset = Multi30k.splits(exts=(".de", ".en"), fields=(self.source, self.target))
        return train_dataset, valid_dataset, test_dataset

    def build_vocab(self, train_dataset, min_freq):
        self.source.build_vocab(train_dataset, min_freq=min_freq)
        self.target.build_vocab(train_dataset, min_freq=min_freq)

    def make_iter(self, train_dataset, valid_dataset, test_dataset, batch_size, device):
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (train_dataset, valid_dataset, test_dataset),
            batch_size=batch_size,
            device=device)

        return train_iterator, valid_iterator, test_iterator