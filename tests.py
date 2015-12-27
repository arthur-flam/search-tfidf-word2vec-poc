import unittest
from tokenizers import MY_TOKENIZER_INDEX

class TokenizerTestCase(unittest.TestCase):
  correct_tokens = [("hello", 1), ("world", 1)]
  def test_format(self):
    """ tokennizer returns [(term,weight),...] format """

  def test_lowercase(self):
    """ tokennizer lowercases """
    self.assertEqual(MY_TOKENIZER_INDEX.tokenize("HELLO"), [("hello",1)])

  def test_split(self):
    """ tokennizer splits on whitespace """
    self.assertEqual(MY_TOKENIZER_INDEX.tokenize("HELLO WORLD"), self.correct_tokens)

  def test_punctuation(self):
    """ tokennizer removes punctuation """
    self.assertEqual(MY_TOKENIZER_INDEX.tokenize("hello world!"), self.correct_tokens)

  def test_stopwords(self):
    """ tokennizer removes stopwords """
    self.assertEqual(MY_TOKENIZER_INDEX.tokenize("hello the world"), self.correct_tokens)
      
  def test_trim_whitespace(self):
    """ tokennizer removes stopwords """
    self.assertEqual(MY_TOKENIZER_INDEX.tokenize("hello   world"), self.correct_tokens)
      
  def test_smallwords(self):
    """ tokennizer removes small words """
    self.assertEqual(MY_TOKENIZER_INDEX.tokenize("hello v world"), self.correct_tokens)

if __name__ == '__main__':
  unittest.main()
