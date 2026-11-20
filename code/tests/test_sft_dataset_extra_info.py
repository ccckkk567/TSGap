import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from warnings import catch_warnings, filterwarnings

import pyarrow as pa
import pyarrow.parquet as pq

from verl.utils.dataset.sft_dataset import SFTDataset


class TestSFTDatasetExtraInfo(unittest.TestCase):
    def test_nested_extra_info(self):
        rows = [
            {"data_source": "alfworld_teaching", "extra_info": {"question": "Q1", "answer": "A1"}},
            {"data_source": "alfworld_teaching", "extra_info": {"question": "Q2", "answer": "A2"}},
        ]
        with TemporaryDirectory() as temp:
            path = Path(temp) / "train.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path)
            with catch_warnings():
                # Pandas 2 warns about the same integer lookup that fails on Orion.
                filterwarnings("error", message="Series.__getitem__ treating keys as positions", category=FutureWarning)
                dataset = SFTDataset(str(path), tokenizer=object(), config={
                    "prompt_key": "extra_info", "prompt_dict_keys": ["question"],
                    "response_key": "extra_info", "response_dict_keys": ["answer"],
                })
            self.assertEqual(dataset.prompts, ["Q1", "Q2"])
            self.assertEqual(dataset.responses, ["A1", "A2"])
