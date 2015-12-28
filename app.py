# -*- coding: utf-8 -*-
"""
Answers queries about documents with tf-idf
"""
import json
from flask import Flask, request, render_template
from term_document import TermDocumentMatrixDD
from tokenizers import MY_TOKENIZER_INDEX, MY_TOKENIZER_SEARCH, MY_TOKENIZER_NO_STEMMING
from loader import LOADER_MONGODB
import settings


tfidf_vanilla  = TermDocumentMatrixDD(LOADER_MONGODB, MY_TOKENIZER_INDEX, MY_TOKENIZER_INDEX)
tfidf_word2vec = TermDocumentMatrixDD(LOADER_MONGODB, MY_TOKENIZER_INDEX, MY_TOKENIZER_SEARCH) # query expansion with word2vec


app = Flask(__name__)
@app.route("/")
def search():
    query = request.args.get('q', '')
    if not query:
        query = "white wooden baby bedding"
    return render_template('search.html', results=tfidf_vanilla.search(query),
                           text='<p>A search page using word2vec (to provide synonyms+weights) is <a href="/word2vec">available here</a>.</p>')

@app.route("/word2vec")
def search_word2vec():
    query = request.args.get('q', '')
    return render_template('search.html', results=tfidf_word2vec.search(query))

@app.route("/similar")
def similar():
    """ Show some word2vec similar words"""
    query = request.args.get('q', '')
    return render_template('similar.html', results=MY_TOKENIZER_NO_STEMMING.tokenize(query))

if __name__ == "__main__":
    app.run(debug=settings.DEBUG, port=6001)

