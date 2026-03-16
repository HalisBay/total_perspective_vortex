import mne
from mne.datasets import eegbci
from pathlib import Path
import sys
from autoreject import AutoReject


sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml_service.data_layer.data_connector import load_subdataset


class DataProcessor:
    """EEG data processing and filtering class."""

    def band_pass(self, raw, l_freq=1.0, h_freq=40.0):
        """Apply band-pass filter (default 1-40 Hz)."""
        return raw.filter(l_freq=l_freq, h_freq=h_freq)

    def notch_filter(self, raw, freqs=50.0):
        """Apply notch filter (default 50 Hz power line noise)."""
        return raw.notch_filter(freqs=freqs)

    def ICA_filter(self, raw, exclude=None, show_plots=False):
        """Apply Independent Component Analysis for artifact removal."""
        new_names = {ch: ch.replace(".", "") for ch in raw.ch_names}
        raw.rename_channels(new_names)
        raw.set_montage("standard_1005", match_case=False)

        ica = mne.preprocessing.ICA(n_components=0.99, random_state=97, max_iter="auto")
        ica.fit(raw)
        fitted_n_components = int(ica.n_components_)

        if show_plots:
            ica.plot_components()
            ica.plot_sources(raw)

        # If provided, keep only valid indices for this ICA fit.
        if exclude is not None:
            requested = [int(idx) for idx in exclude]
            valid_exclude = [idx for idx in requested if 0 <= idx < fitted_n_components]
            dropped = sorted(set(requested) - set(valid_exclude))
            ica.exclude = valid_exclude

        if show_plots and ica.exclude:
            ica.plot_properties(raw, picks=ica.exclude)

        return ica.apply(raw, exclude=ica.exclude)

    def clean_eeg_data(self, raw, exclude):
        filtered_raw = self.band_pass(raw)
        filtered_raw = self.notch_filter(filtered_raw)
        filtered_raw = self.ICA_filter(
            filtered_raw,
            exclude=exclude,  # 8,11,15
            show_plots=False,
        )
        return filtered_raw

    def find_events(self, raw):
        events, event_id = mne.events_from_annotations(raw)
        # print(events)
        # print(event_id)
        return events, event_id

    def create_epochs(self, raw, events, event_id):

        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=1,
            tmax=4,
            baseline=None,
            preload=True,
        )
        epochs = epochs["T1", "T2"]
        ar = AutoReject()
        nr_epochs = ar.fit_transform(epochs)
        print(nr_epochs)
        print(nr_epochs.get_data().shape)
        return nr_epochs

    def EDA(self, raw):
        """Exploratory Data Analysis - print raw data information."""
        print(raw)
        print(raw.info)


if __name__ == "__main__":
    raws = load_subdataset()
    raw = raws[0]
    processor = DataProcessor()
    processor.EDA(raw)
    # raw = processor.band_pass(raw)
    # raw = processor.notch_filter(raw)
    # processor.ICA_filter(raw)
