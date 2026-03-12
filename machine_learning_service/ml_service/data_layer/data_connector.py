import mne
from mne.datasets import eegbci
from pathlib import Path

"""
In summary, the experimental runs were:
Baseline, eyes open
Baseline, eyes closed
Task 1 (open and close left or right fist)
Task 2 (imagine opening and closing left or right fist)
Task 3 (open and close both fists or both feet)
Task 4 (imagine opening and closing both fists or both feet)
Task 1
Task 2
Task 3
Task 4
Task 1
Task 2
Task 3
Task 4

"""


base = Path(__file__).parents[2]  # machine_learning_service/
data_path = base / "data"
data_path.mkdir(parents=True, exist_ok=True)


def load_subdataset():
    """Download a small pilot subset into `machine_learning_service/data/` and return Raw objects."""
    subjects = range(1, 6)
    runs = [6, 10, 14]  # motor imagery
    files: list[str] = []

    for s in subjects:
        files += eegbci.load_data(s, runs, path=str(data_path))

    raws = [mne.io.read_raw_edf(f, preload=True) for f in files]
    print(f"Loaded {len(raws)} raws; files saved to: {data_path}")


    return raws
