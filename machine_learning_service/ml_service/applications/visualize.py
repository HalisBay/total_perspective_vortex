import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_service.data_layer.data_connector import load_subdataset
from ml_service.machine_learning.data_processor import DataProcessor

processor = DataProcessor()


def ensure_channel_locations(raw):
    """Set a standard montage so PSD can use spatial channel colors."""

    new_names = {ch: ch.replace(".", "") for ch in raw.ch_names}
    raw.rename_channels(new_names)
    raw.set_montage("standard_1005", match_case=False)


def plotter(raw, title, show_trace=False):
    """Plot PSD for Raw/Epochs and optionally plot traces."""
    ensure_channel_locations(raw)

    if show_trace:
        if hasattr(raw, "events"):
            raw.plot(n_epochs=10, n_channels=30, title=title)
        else:
            raw.plot(duration=5, n_channels=30, title=title)

    fig = raw.compute_psd(fmax=50).plot(
        picks="data", exclude="bads", amplitude=False, spatial_colors=True, show=False
    )
    fig.axes[0].set_title(title)
    plt.show()


def filter_plotter(raw, filter_raw):
    raw.plot(duration=5, n_channels=30, title="Before Filtering EEG")
    filter_raw.plot(duration=5, n_channels=30, title="After Filtering EEG")
    plt.show()


if __name__ == "__main__":
    raws = load_subdataset()
    raws = list(raws)
    raw = raws[0]
    raw_original = raw.copy()
    #   plotter(raw)
    filter_raw = processor.clean_eeg_data(raw, [0, 2, 4, 5, 9, 8, 10, 16])
    # filter_plotter(raw_original, filter_raw)

    events, e_id = processor.find_events(filter_raw)
    epochs = processor.create_epochs(filter_raw, events, e_id)
    plotter(raw_original, show_trace=False, title="Original EEG")
    plotter(epochs, show_trace=False, title="Filtered EEG")
