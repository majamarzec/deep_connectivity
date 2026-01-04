# src/preprocessing/processing.py

from typing import Optional
import mne
from mne.io import BaseRaw
import numpy as np
from scipy.signal import iirnotch, butter

from pathlib import Path

print(Path(__file__).resolve())

from .constants import (
    BASE_DIR, BASE_CSV_PATH, EDF_DIR,
    CH_NAMES, VALID_1020, CHNAMES_MAPPING,
    FREQ_BANDS, FREQ_BANDS_NAMES,
    DEFAULT_NOTCH_FREQ, DEFAULT_NOTCH_Q,
    DEFAULT_HP_CUTOFF, DEFAULT_LP_CUTOFF, DEFAULT_MONTAGE,
    DEFAULT_SFREQ, DEFAULT_PERCENTILE, DEFAULT_IIR_ORDER
)

from .utils import apply_mor_data_hack_fix, chunk


class EEGPreprocessor:
    """
    Main EEG preprocessing pipeline (EDF → standardized channels → filters → rereference...).
    """

    def __init__(self, raw: mne.io.BaseRaw):
        self.raw = raw

    # ============ LOADER ============
    @classmethod
    def from_edf(cls, edf_path: str, institution_id: Optional[str] = None, preload=True):
        raw = mne.io.read_raw_edf(edf_path, preload=preload)
        if institution_id is not None:
            raw = apply_mor_data_hack_fix(raw, edf_path, institution_id) 
        return cls(raw)

    # ============ CHANNELS ============
    def standardize_channels(
        self,
        valid_channels=VALID_1020,
        channel_mappings=CHNAMES_MAPPING,
        target_order=CH_NAMES,
        plot_montage=False,
        inplace=False
    ):
        raw = self.raw if inplace else self.raw.copy()
        keep = [ch for ch in raw.ch_names if ch in valid_channels]
        raw.pick_channels(keep)

        raw_ch_set = set(raw.ch_names)
        match_ratio = len(raw_ch_set.intersection(valid_channels)) / len(raw_ch_set)

        if match_ratio < 0.8:
            best_mapping = None
            best_score = -1
            for mapping in channel_mappings:
                score = len(raw_ch_set.intersection(mapping.keys()))
                if score > best_score:
                    best_score = score
                    best_mapping = mapping
            if best_score >= 3:
                raw.rename_channels(best_mapping)

        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore")
        if plot_montage:
            montage.plot(sphere=0.1)

        ordered = [ch for ch in target_order if ch in raw.ch_names]
        raw.reorder_channels(ordered)

        if inplace:
            self.raw = raw
            return self
        return raw

   # ============ FILTERING ============
    def apply_filters(
        self,
        notch_freq=DEFAULT_NOTCH_FREQ,
        notch_q=DEFAULT_NOTCH_Q,
        hp_cutoff=DEFAULT_HP_CUTOFF,
        lp_cutoff=DEFAULT_LP_CUTOFF,
        order=DEFAULT_IIR_ORDER,
        inplace=False,
        fs=None,
    ):
        """
        Use MNE's built-in filters (most stable approach)
        """
        raw = self.raw if inplace else self.raw.copy()
        
        # 1) NOTCH - use MNE's built-in
        if notch_freq is not None:
            raw.notch_filter(freqs=notch_freq, picks='data', verbose=False)
        
        # 2) BANDPASS - use MNE's built-in (it handles HP and LP together)
        if hp_cutoff is not None or lp_cutoff is not None:
            raw.filter(
                l_freq=hp_cutoff, 
                h_freq=lp_cutoff, 
                picks='data',
                method='iir',
                iir_params={'order': order, 'ftype': 'butter'},
                verbose=False
            )
        
        if inplace:
            self.raw = raw
            return self
            
        return raw
    # ============ REFERENCE & RESAMPLE ============
    def rereference(self, ref_channels=DEFAULT_MONTAGE):
        self.raw.set_eeg_reference(ref_channels)
        return self

    def resample(self, sfreq=DEFAULT_SFREQ):
        if self.raw.info["sfreq"] != sfreq:
            self.raw.resample(float(sfreq))
        return self

    # ============ FULL PIPELINE ============
    def preprocess(
        self,
        ref_channels=DEFAULT_MONTAGE,
        sfreq=DEFAULT_SFREQ,
        notch_freq=DEFAULT_NOTCH_FREQ,
        hp_cutoff=DEFAULT_HP_CUTOFF,
        lp_cutoff=DEFAULT_LP_CUTOFF,
        plot=False,
        percentile=DEFAULT_PERCENTILE
    ):
        self.standardize_channels(inplace=True)
        self.resample(sfreq)
        self.apply_filters(
            notch_freq=notch_freq,
            hp_cutoff=hp_cutoff,
            lp_cutoff=lp_cutoff,
            order=DEFAULT_IIR_ORDER,
            inplace=True,
            fs=sfreq
        )
        self.rereference(ref_channels)

        if plot:
            scaling = {"eeg": np.percentile(np.abs(self.raw.get_data()), percentile)}
            self.raw.plot(scalings=scaling)

        return self.raw