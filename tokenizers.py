# -*- coding: utf-8 -*-
"""
Util to tokenize strings in an extensible manner
"""
import re
from nltk.stem.porter import PorterStemmer
#from nltk.corpus import stopwords
stemmer = PorterStemmer()

class Tokenizer(object):
    """ Handles tokenizer function """
    def __init__(self, transforms=[], filters=[]):
        self.transforms = transforms # list of string -> string
        self.filters = filters       # list of tokens -> tokens

    def tokenize(self, string):
        s = string
        for transform in self.transforms:
            s = transform(s)
        tokens = s.split(' ')
        for filter in self.filters:
            tokens = [token for token in tokens if not filter(token)]
        return tokens

transforms = [
    lambda s: s.lower(),
    lambda s: re.sub('[^A-Za-z0-9 ]+', '', s)
]
filters = [
    lambda s: len(s) < 4,
    lambda s: s in ["stopwords", "the", "a", "stop", "what"]
]

my_tokenizer = Tokenizer(transforms, filters)
