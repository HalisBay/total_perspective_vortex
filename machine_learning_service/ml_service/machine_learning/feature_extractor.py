import mne
from mne.time_frequency.psd import psd_array_welch
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml_service.data_layer.data_connector import load_subdataset
from ml_service.machine_learning.data_processor import DataProcessor


class FeatureExtractor:
    def __init__(self):
        self.bandwidth = [
            (0.5, 3.5),
            (4, 7.5),
            (8, 12),
            (13, 35),
        ]

    def set_freqs(self, freqs, psds, min_hz, max_hz):
        # build boolean mask correctly for arrays
        w_freqs = (freqs >= min_hz) & (freqs <= max_hz)
        output = psds[:, :, w_freqs].mean(axis=2)
        return output

    def transformer(self, epochs):

        psds, freqs = psd_array_welch(
            epochs.get_data(),
            epochs.info.get("sfreq", None),
            fmin=0.5,
            fmax=40,
            n_fft=256,
        )
        features = []
        for min_hz, max_hz in self.bandwidth:
            f_bandwith = self.set_freqs(freqs, psds, min_hz, max_hz)
            features.append(f_bandwith)
        return np.concatenate(features, axis=1)


processor = DataProcessor()
extractor = FeatureExtractor()

if __name__ == "__main__":
    # data connector
    raws = load_subdataset()
    raws = list(raws)
    raw = raws[0]
    raw_original = raw.copy()

    # data process
    filter_raw = processor.clean_eeg_data(raw, [0, 2, 4, 5, 9, 8, 10, 16])
    events, e_id = processor.find_events(filter_raw)
    epochs = processor.create_epochs(filter_raw, events, e_id)

    # feature extract (signal process)

    data = epochs.get_data()  # shape (n_epochs, n_channels, n_times)
    sfreq = epochs.info.get("sfreq", None)

    # print("freqs : ", freqs)
    # print("psds : " ,psds)
    all_shapes = extractor.transformer(epochs)  # shape (n_epochs, n_channels)
    print(all_shapes.shape)
    # print(psds.shape)
    # print(freqs.shape)
