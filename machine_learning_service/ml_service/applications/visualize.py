import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_service.data_layer.data_connector import load_subdataset
from ml_service.machine_learning.data_processor import DataProcessor

processor = DataProcessor()


def plotter(raw):
    """Plot raw EEG data."""
    raw.compute_psd(fmax=50).plot(picks="data", exclude="bads", amplitude=False)
    raw.plot(duration=5, n_channels=30)
    plt.show()


def filter_plotter(raw):
    raw_original = raw.copy()

    filtered_raw = processor.band_pass(raw)
    filtered_raw = processor.notch_filter(filtered_raw)
    filtered_raw = processor.ICA_filter(
        filtered_raw,
        exclude=[0, 2, 4, 5, 9, 8, 10, 16], #8,11,15
        show_plots=False,
    )
    raw_original.plot(duration=5, n_channels=30, title="Before Filtering EEG")
    filtered_raw.plot(
        duration=5, n_channels=30, title="After Filtering EEG"
    )
    plt.show()


if __name__ == "__main__":
    raws = load_subdataset()
    raws = list(raws)
    raw = raws[0]
    # plotter(raw)

    filter_plotter(raw)
