# -*- coding: utf-8 -*-
"""
Data structure to hold corpus data and answer tf-idf queries
"""
from abc import ABCMeta, abstractmethod
import math
import operator


class TermDocumentMatrixAbstract:
    """
    Data structure holding the corpus and frequency data.
    Needs 'backend' Implementation.
    """
    __metaclass__ = ABCMeta
    @abstractmethod
    def get_freq(self, doc_id, term): pass
    @abstractmethod
    def set_freq(self, doc_id, term, value): pass
    @abstractmethod
    def get_corpus_freq(self, term): pass
    @abstractmethod
    def set_corpus_freq(self, term, value): pass
    @abstractmethod
    def get_terms(): pass
    @abstractmethod
    def get_documents_for_term(self, term):
        """ get an array of ids for document matching the terms """
        pass
    @abstractmethod
    def add_document(self, doc_id):
        """ allocate data for document doc_id """
        pass

    def add_new_document(self, doc_id, terms):
        self.add_document(doc_id)
        for term, weight in terms:
            if weight == 1: # we only add exact terms to the index
                old_corpus_freq = self.get_corpus_freq(term)
                self.set_corpus_freq(term, old_corpus_freq + 1)
                self.set_freq(doc_id, term, self.get_freq(doc_id, term) + 1)
    def perform_tf_idf(self):
        for term in self.get_terms():
            inverse_term_frequency = math.log(
                self.n_documents / self.counts_of_words_in_corpus[term]
            )
            matching_documents = self.get_documents_for_term(term)
            for doc_id in matching_documents:
                old_term_freq = self.get_freq(doc_id, term)
                # note: here we define freq as number of occurences
                # - as descriptions have ~same amount of words
                # - easier to get and update
                self.set_freq(doc_id, term, inverse_term_frequency * old_term_freq)

    def __init__(self, documents_loader, tokenizer_index, tokenizer_search):
        self.n_documents = 0
        self.tokenizer_index = tokenizer_index
        self.tokenizer_search = tokenizer_search
        self.documents = dict()
        for document in documents_loader():
            self.documents[document["Id"]] = document
            doc_id = document["Id"]
            terms = self.tokenizer_index.tokenize(document["Name"])
            self.add_new_document(doc_id, terms)
            self.n_documents = self.n_documents + 1
        print("Loaded " + str(self.n_documents) + " document")
        self.perform_tf_idf()

    def search(self, query, n=10):
        """ Search the corpus using the tokenizer and tf-idf """
        query_terms = self.tokenizer_search.tokenize(query)
        scores = dict()
        print(query_terms)
        for term, weight in query_terms:
            matching_documents = self.get_documents_for_term(term)
            for doc_id in matching_documents:
                if doc_id not in scores:
                    scores[doc_id] = 0
                scores[doc_id] = scores[doc_id] + self.get_freq(doc_id, term) * weight
        scores_sorted = sorted(scores.items(), key=operator.itemgetter(1, 0), reverse=True)
        best_scores = list(scores_sorted)[0:(n-1)]
        output = [{
            "Id":document[0],
            "product":self.documents[document[0]],
            "score":document[1]
            } for document in best_scores]
        return output


class TermDocumentMatrixDD(TermDocumentMatrixAbstract):
    """ Implementation with dict-of-dicts """
    counts_of_words_in_corpus = dict()
    term_document_frequencies = dict()
    documents_for_term = dict()

    def get_terms(self):
        return list(self.documents_for_term.keys())
    def add_document(self, doc_id):
        if doc_id not in self.term_document_frequencies:
            self.term_document_frequencies[doc_id] = dict()
    def get_freq(self, doc_id, term):
        if term not in self.term_document_frequencies[doc_id]:
            return 0
        return self.term_document_frequencies[doc_id][term]
    def set_freq(self, doc_id, term, value):
        if term not in self.term_document_frequencies[doc_id]:
            self.term_document_frequencies[doc_id][term] = 0
        self.term_document_frequencies[doc_id][term] = value
        if term not in self.documents_for_term:
            self.documents_for_term[term] = set()
        self.documents_for_term[term].add(doc_id) # we assume we set != 0
        return value
    def get_corpus_freq(self, term):
        if term not in self.counts_of_words_in_corpus:
            self.counts_of_words_in_corpus[term] = 0
        return self.counts_of_words_in_corpus[term]
    def set_corpus_freq(self, term, value):
        if term not in self.counts_of_words_in_corpus:
            self.counts_of_words_in_corpus[term] = 0
        self.counts_of_words_in_corpus[term] = value
        return value
    def get_documents_for_term(self, term):
        if term not in self.documents_for_term:
            return set()
        return self.documents_for_term[term]

class TermDocumentMatrixSM(TermDocumentMatrixAbstract):
    """ Implementation with sparse matrix """
    pass
