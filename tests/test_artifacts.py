import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SelectedArtifactTests(unittest.TestCase):
    def test_shipped_splits_match_report_and_exclude_regression_controls(self):
        processed = json.loads((ROOT / 'data/processed/dataset.json').read_text(encoding='utf-8'))
        test = json.loads((ROOT / 'data/test/test.json').read_text(encoding='utf-8'))['test']
        report = json.loads((ROOT / 'results/curation-selected/report.json').read_text(encoding='utf-8'))
        self.assertEqual((len(processed['train']), len(processed['validation']), len(test)), (6000, 500, 500))
        indices = report['source_indices']['train'] + report['source_indices']['validation'] + report['source_indices']['test']
        self.assertEqual(len(indices), 7000)
        self.assertEqual(len(set(indices)), 7000)
        self.assertTrue({58, 185, 187, 7565, 14279, 15291}.isdisjoint(indices))
        examples = processed['train'] + processed['validation'] + test
        keys = {(row['instruction'], row['input'], row['output']) for row in examples}
        self.assertEqual(len(keys), 7000)


if __name__ == '__main__':
    unittest.main()
