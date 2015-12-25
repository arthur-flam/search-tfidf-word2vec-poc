# Searching documents with TF-IDF & Word2vec

Demo should be running at [search.lookies.io](http://search.lookies.io?purple+chair).
![Preview on test data](https://raw.github.com/arthur-flam/tf-idf-poc/master/screenshot.png)

## Setup
```
# sudo apt-get install python3.4-dev
virtualenv -p /usr/local/bin/python3 py3env # see: which python3
source py3env/bin/activate
pip install Flask pymongo
pip install nltk    # for the stemmer (todo)

pip install gensim  # for word2vec # cython numpy word2vec
wget https://s3.amazonaws.com/mordecai-geo/GoogleNews-vectors-negative300.bin.gz
# mirror of: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
```

## Run
```
python app.py
```

## Tests
I clearly need to transition from `build then test/maintain` to `test then build/maintain`...
- The tokenizer is tested.
- The rest *still* not so much...

## Organisation and classes

### Term-document data structure
- All the logic is in the class `term_document_matrix_abstract` in *term_document.py*
- Low-level details are drafted in abstract methods and left to be implemented
- A implementation using a **dict-of-dicts** is available.
- A sparse matrix could be usefull as well and *interesting for a comparaison*.

### Tokenizer
- String transforms and token filters can be used easily to create a `tokenizer`
- Stemming, lowercasing and some others are shown as an example in *tokenizers.py*
- Tokenization is available through 
```
tokens = my_tokenizer.tokenize(string)
```

### Data loading
- **Extensible** : we just need to provide the `term_document_matrix_abstract` constructor with an iterable over documents
- Available : a JSON file reader and one fetching docs from MongoDB.
- We could read the JSON in chunks but since we are keeping all the data in memory anyway...

### Web GUI
Run and it is here : [http://localhost:5000?q=macbook](http://localhost:5000?q=macbook).

## What is bad
- Maybe more [SOLID](https://en.wikipedia.org/wiki/SOLID_(object-oriented_design)) to have the term-doc-freq data structure as member of the main data structure
- Documents are handled as {"Id":"XXXX", "Name":"text"}. More flexibility could help.
- Typing is poor (Python..)
- Python3 only.

## Todo
- Complete stemming
- Rewrite our use of word2vec as query expansion

## Performance
* Indexing : Time should grow in *O(tokens) ~= O(documents)* 
* Index size : the dict-of-dicts approach is heavy...
* Search : *O(n * ln(n))* where *n* is the number of documents where the query terms appear.
