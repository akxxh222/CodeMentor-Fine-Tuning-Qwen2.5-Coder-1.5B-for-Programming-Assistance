import os
import unittest

from src.local_review import load_local_model


class LocalReviewIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("CODEMENTOR_TEST_MODEL"), "Set CODEMENTOR_TEST_MODEL to a local snapshot")
    def test_transformers_loader_places_cached_quantized_model_on_cuda(self):
        model, tokenizer = load_local_model(os.environ["CODEMENTOR_TEST_MODEL"])
        self.assertEqual(next(model.parameters()).device.type, "cuda")
        self.assertIsNotNone(tokenizer.chat_template)


if __name__ == "__main__":
    unittest.main()
