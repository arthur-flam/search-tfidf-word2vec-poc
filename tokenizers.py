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
    def __init__(self, transforms=None, filters=None):
        self._transforms = transforms # list of string -> string
        self._filters = filters       # list of tokens -> tokens

    def apply_transforms(self, string):
        """ turns a raw string into an processed string """
        for transform in self._transforms:
            string = transform(string)
        return string

    def apply_filters(self, tokens):
        """ filters an array of tokens """
        for filter_ in self._filters:
            tokens = [token for token in tokens if filter_(token)]
        return tokens

    def tokenize(self, string):
        """ turns a raw string into an array of tokens """
        tokens = self.apply_transforms(string).split(' ')
        return self.apply_filters(tokens)

my_transforms = [
    lambda s: s.lower(),
    lambda s: re.sub('[^A-Za-z0-9 ]+', '', s)
]
my_filters = [
    lambda s: len(s) > 3,
    lambda s: s not in ["stopwords", "the", "a", "stop", "what"]
]

my_tokenizer = Tokenizer(my_transforms, my_filters)
