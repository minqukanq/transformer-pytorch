import spacy


class Tokenizer:
    def __init__(self):
        self.spacy_en = spacy.load('en')
        self.spacy_de = spacy.load('de')

    def tokenize_de(self, text):
        return [token.text for token in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        return [token.text for token in self.spacy_en.tokenizer(text)]