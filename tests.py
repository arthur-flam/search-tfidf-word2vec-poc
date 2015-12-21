import unittest

from tokenizers import my_tokenizer
class TokenizerTestCase(unittest.TestCase):
  def test_basic(self):
      self.assertEqual(my_tokenizer.tokenize("HELLO"), ["hello"])
      self.assertEqual(my_tokenizer.tokenize("HELLO WORLD"), ["hello","world"])
      self.assertEqual(my_tokenizer.tokenize("hello world!"), ["hello","world"])
      self.assertEqual(my_tokenizer.tokenize("hello the world"), ["hello","world"])
      self.assertEqual(my_tokenizer.tokenize("hello   world"), ["hello","world"])
      self.assertEqual(my_tokenizer.tokenize("hello v world"), ["hello","world"])

# import os
# import app
# import unittest
# import tempfile
# class FlaskAppTestCase(unittest.TestCase):

#     def setUp(self):
#         self.app = app.app.test_client()
#         print(app)

#     def test_empty_db(self):
#         rv = self.app.get('/search')
#         assert 'Search!' in rv.data

# if __name__ == '__main__':
#     unittest.main()