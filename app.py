# -*- coding: utf-8 -*-
"""
Answers queries about documents with tf-idf
"""
import json
from flask import Flask, request, render_template
from term_document import TermDocumentMatrixDD
from tokenizers import my_tokenizer


# ===========================================================================
# TODO :
# - a class for documents will enable doc.get_text() rather than doc["Name"]
# ===========================================================================

def json_loader(filepath="products.json"):
    """
    Returns an array of documents read from the JSON file
    Other function will work as long as they return iterable types
    """
    file_connection = open(filepath, 'r', encoding="utf-8")
    return json.loads(file_connection.read(), encoding="utf-8")

# == INIT ===================================================================
tfidf_data = TermDocumentMatrixDD(json_loader, my_tokenizer)


# === SERVER ================================================================
app = Flask(__name__)
@app.route("/")
def search():
    query = request.args.get('q', '')
    return render_template('search.html', results=tfidf_data.search(query))

if __name__ == "__main__":
    app.run(debug=True)
