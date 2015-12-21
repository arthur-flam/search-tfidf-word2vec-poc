# tfidf

## Setup
```
virtualenv -p /usr/local/bin/python3 py3env # see: which python3
source py3env/bin/activate
pip install Flask nltk
```

## Run
```
python3 app.py
```

## Tests
- The tokenizer is tested
- The rest not so much still.
- Will need to transition from `build then test/maintain` to `test then build/maintain`...

## Organisation and classes

### Term-document data structure
- All the logic is in the class `term_document_matrix_abstract` in *term_document.py*
- Low-level details are left to be implemented in abstract methods
- A implementation using a dict-of-dicts is available. A sparse matrix could be usefull as well

### Tokenizer
- String transforms and token filters can be used easily to create a `tokenizer`
- Stemming, lowercasing and some others are shown as an example in *tokenizers.py*
- Tokenization is available through 
```
tokens = my_tokenizer.tokenize(string)
```

### Data loading
- Extensible : we just need to provide the `term_document_matrix_abstract` constructor with an iterable over documents
- Available : a JSON file reader.
- We could read the JSON in chunks but since we are keeping all the data in memory anyway...

### Web GUI
Run and it is here : [http://localhost:5000?q=macbook](http://localhost:5000?q=macbook).

### What is bad
- Documents are handled as {"Id":"XXXX", "Name":"text"}. More flexibility could help
- Typing is poor (Python..)
- Python3 only.

### Performance
* Indexing : Time should grow in *O(tokens) ~= O(documents)* 
* Index size : the dict-of-dicts approach is heavy...
* Search : *O(n * ln(n))* where *n* is the number of documents where the query terms appear.
