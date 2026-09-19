"""Local integration regression for Transformers chat-template return changes."""
import os
from pathlib import Path
import unittest
from src.curation import conversation_tokens


class TokenizationTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('CODEMENTOR_TEST_TOKENIZER'), 'Set CODEMENTOR_TEST_TOKENIZER to a local snapshot')
    def test_real_chat_length_counts_ids_not_dictionary_keys(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(Path(os.environ['CODEMENTOR_TEST_TOKENIZER'])), local_files_only=True)
        short = dict(instruction='Explain a Python list.', input='', output='A list holds items.')
        long = dict(short, output=' '.join(['variable'] * 1000))
        self.assertGreater(conversation_tokens(short, tokenizer), 10)
        self.assertGreater(conversation_tokens(long, tokenizer), 512)
        self.assertGreater(conversation_tokens(long, tokenizer), conversation_tokens(short, tokenizer))


if __name__ == '__main__':
    unittest.main()
