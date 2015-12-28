# -*- coding: utf-8 -*-
"""
Util to tokenize strings in an extensible manner
Will return an array of tuples [(term, weight)]
"""
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import gensim

class Tokenizer(object):
    """
    Handles tokenizer function.
    Will split on whitespace.
    """
    def __init__(self, string_transforms=None, array_transforms=None, filters=None):
        self._string_transforms = string_transforms # string -> string
        self._filters = filters                     # list of tokens -> subset(list of tokens)
        self._array_transforms = array_transforms   # list of tokens -> list of tokens

    def apply_string_transforms(self, string):
        """ Turns a raw string into an processed string """
        for transform in self._string_transforms:
            string = transform(string)
        return string

    def apply_array_transforms(self, tokens):
        """ Turns tokens into other tokens """
        for transform in self._array_transforms:
            tokens = transform(tokens)
        return tokens

    def apply_filters(self, tokens):
        """ Filters an array of tokens """
        for filter_ in self._filters:
            tokens = [(token, weight) for token, weight in tokens if filter_(token)]
        return tokens

    def tokenize(self, string):
        """ turns a raw string into an array of tokens """
        tokens = self.apply_string_transforms(string).split(' ')
        tokens = [(token, 1) for token in tokens]
        tokens = self.apply_filters(tokens)
        tokens = self.apply_array_transforms(tokens)
        return tokens



stemmer = PorterStemmer()
word2vec = gensim.models.Word2Vec.load("custom_word2vec.bin")

def query_expansion_word2vec(tokens):
    """
    Query expansion using our word2vec dictionnary seen as a 'synonyms' provider.
    See https://radimrehurek.com/gensim/models/word2vec.html
    """
    try:
        similar_tokens_array_of_arrays = [word2vec.most_similar(positive=[token], topn=10) for token in tokens]
    except KeyError:
        similar_tokens_array_of_arrays = []
    similar_tokens_array = [token for sublist in similar_tokens_array_of_arrays for token in sublist]
    return tokens + similar_tokens_array

MY_STRING_TRANSFORMS = [
    lambda s: s.lower(),
    lambda s: re.sub('[^A-Za-z0-9 ]+', '', s)
]
MY_ARRAY_TRANSFORMS_INDEX = [
    lambda tokens: list(set(tokens)), # remove duplicates to help with "cheating",
    lambda tokens: [(stemmer.stem(token), weight) for token, weight in tokens]
]
MY_ARRAY_TRANSFORMS_SEARCH = [
    query_expansion_word2vec, # done before stemming, so we count on word2vec...
    lambda tokens: [(stemmer.stem(token), weight) for token, weight in tokens]
]
MY_FILTERS = [
    lambda s: len(s) > 3,
    lambda s: s not in ["stopwords", "the", "a", "stop", "what"]
]

MY_TOKENIZER_INDEX  = Tokenizer(MY_STRING_TRANSFORMS, MY_ARRAY_TRANSFORMS_INDEX, MY_FILTERS) 
MY_TOKENIZER_SEARCH = Tokenizer(MY_STRING_TRANSFORMS, MY_ARRAY_TRANSFORMS_SEARCH, MY_FILTERS)
MY_TOKENIZER_NO_STEMMING = Tokenizer(MY_STRING_TRANSFORMS, [query_expansion_word2vec], MY_FILTERS)

