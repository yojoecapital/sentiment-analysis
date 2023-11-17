import spacy
import torch
from torch.utils.data import Dataset
import pandas as pd
from nltk.stem.snowball  import SnowballStemmer
from torchtext.vocab import build_vocab_from_iterator
import torchtext.transforms as T
import torch.nn.functional as F

class SentimentDataset(Dataset):
    def __init__(self, path, fix_length=32, min_freq=2, max_tokens=25_000, count=None, seed=20, device="cpu"):

        self.fix_length = fix_length
        self.device = device
        if self.device is str:
            self.device = torch.device(device)

        self.df = pd.read_csv(path, header=None, encoding='latin-1')
        self.df = self.df.sample(frac=1.0, random_state=seed)

        if (count is not None):
            self.df = self.df.head(count)
            
        self.x = self.df[5].to_numpy()
        self.y = self.df[0].apply(lambda x: 1.0 if x != 0 else 0.0)
        self.y = torch.from_numpy(self.y.to_numpy()).to(self.device)

        self.eng = spacy.load("en_core_web_sm") 
        self.stemmer = SnowballStemmer("english", ignore_stopwords=True)

        self.vocab = build_vocab_from_iterator(
            self.preprocessed_tokens_iter(self.x),
            min_freq=min_freq,

            # special tokens include passing, start & end of sentence, & unknown
            specials=['<pad>', '<sos>', '<eos>', '<unk>'],
            special_first=True,

            max_tokens=max_tokens
        )
        self.vocab.set_default_index(self.vocab['<unk>'])
        self.vocab_size = len(self.vocab)

        
        self.text_tranform = T.Sequential(
            # converts the sequence to indices based on given vocabulary
            T.VocabTransform(vocab=self.vocab),
            
            # add <sos> at beginning of each sentence. 1 because the index for <sos> in vocabulary is
            # 1 because index for <sos> in vocab
            T.AddToken(1, begin=True),

            # add <eos> at beginning of each sentence
            # 2 because index for <eos> in vocab
            T.AddToken(2, begin=False)
        )

    def __getitem__(self, index):
        x = self.text_tranform(self.get_preprocessed_tokens(self.x[index]))
        x = torch.tensor(x)
        x = T.PadTransform(max_length=self.fix_length, pad_value=0)(x)[:self.fix_length].to(self.device)
        assert x.shape == torch.Size([self.fix_length])
        return x, self.y[index]
    
    def __len__(self):
        return len(self.y)

    def get_preprocessed_tokens(self, text):
        """
        Tokenize a text & preprocess each token using preprocess (function) 
        and return a list of tokens
        """
        return [self.stemmer.stem(token.text) for token in self.eng.tokenizer(text)]
    
    def preprocessed_tokens_iter(self, iter):
        for text in iter:
            yield self.get_preprocessed_tokens(text)