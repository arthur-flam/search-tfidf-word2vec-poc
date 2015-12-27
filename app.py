# -*- coding: utf-8 -*-
"""
Answers queries about documents with tf-idf
"""
import json
from flask import Flask, request, render_template
from term_document import TermDocumentMatrixDD
from tokenizers import MY_TOKENIZER_INDEX, MY_TOKENIZER_SEARCH
from loader import LOADER_MONGODB
import settings


tfidf_vanilla  = TermDocumentMatrixDD(LOADER_MONGODB, MY_TOKENIZER_INDEX, MY_TOKENIZER_INDEX)
tfidf_word2vec = TermDocumentMatrixDD(LOADER_MONGODB, MY_TOKENIZER_INDEX, MY_TOKENIZER_SEARCH) # query expansion with word2vec


app = Flask(__name__)
@app.route("/")
def search():
    query = request.args.get('q', '')
    return render_template('search.html', results=tfidf_vanilla.search(query),
                           text='<p>Search using word2vec embeddings (trained on this data) to find "synonyms" and cosine distance as extra weight is <a href="/word2vec">available here</a>.</p>')

@app.route("/word2vec")
def search_word2vec():
    query = request.args.get('q', '')
    return render_template('search.html', results=tfidf_word2vec.search(query))



if __name__ == "__main__":
    app.run(debug=True, port=6001)
    # app.run(port=6001)
