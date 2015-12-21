# -*- coding: utf-8 -*-
import json, codecs, sys
from term_document import term_document_matrix_ll
from tokenizers import my_tokenizer


# ===========================================================================

# ===========================================================================
# TODO : a class for documents
# - will enable get_text rather than doc["Name"]
# ===========================================================================

def json_loader(filepath = "products.json"):
    """ returns an array of documents """
    """ it would be better if it returned an iterator (large files / DB) """
    file = open(filepath,'r', encoding="utf-8")
    return json.loads(file.read(), encoding="utf-8")

# ===========================================================================

tfidf_data = term_document_matrix_ll(json_loader, my_tokenizer)


from flask import Flask, request, render_template
app = Flask(__name__)
@app.route("/")
def search():
    query = request.args.get('q', '')
    return render_template('search.html', results=tfidf_data.search(query))


if __name__ == "__main__":
    app.run(debug=True)