import tempfile
import unittest
from pathlib import Path

from scipy.io import savemat

from datasets.dataset_loader import DatasetLoader


class DatasetLoaderTests(unittest.TestCase):
    def test_reports_missing_directory(self) -> None:
        loader = DatasetLoader()

        with self.assertRaisesRegex(FileNotFoundError, "Data directory not found"):
            loader.load_datasets("/path/does/not/exist")

    def test_loads_summaries_for_existing_mat_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data" / "raw"
            data_dir.mkdir(parents=True)
            mat_path = data_dir / "sample.mat"
            savemat(mat_path, {"signal": [1, 2, 3]})

            loader = DatasetLoader()
            summaries = loader.load_datasets(str(data_dir))

            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["status"], "ok")
            self.assertEqual(
                Path(summaries[0]["path"]).resolve(),
                mat_path.resolve(),
            )
            self.assertIn("signal", summaries[0]["keys"])


if __name__ == "__main__":
    unittest.main()