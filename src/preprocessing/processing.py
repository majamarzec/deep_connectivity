# src/preprocessing/processing.py

from typing import Optional
import mne
import numpy as np
from scipy.signal import iirnotch, butter



from preprocessing.constants import (
    BASE_DIR, BASE_CSV_PATH, EDF_DIR,
    CH_NAMES, VALID_1020, CHNAMES_MAPPING,
    FREQ_BANDS, FREQ_BANDS_NAMES,
    DEFAULT_NOTCH_FREQ, DEFAULT_NOTCH_Q,
    DEFAULT_HP_CUTOFF, DEFAULT_LP_CUTOFF, DEFAULT_MONTAGE,
    DEFAULT_SFREQ, DEFAULT_PERCENTILE, DEFAULT_IIR_ORDER
)

from preprocessing.utils import apply_mor_data_hack_fix


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
        inplace=False,
        fs=None,
    ):
        """
        Apply filters using scipy with proper order calculation for MVAR stability.
        
        Uses:
        - scipy.signal.iirnotch for notch filter
        - scipy.signal.buttord for optimal filter order calculation
        - scipy.signal.butter with SOS (second-order sections) for numerical stability
        
        Filter order is limited to max 4 for MVAR model stability.
        """
        from scipy.signal import buttord
        
        raw = self.raw if inplace else self.raw.copy()
        sfreq = fs if fs is not None else raw.info['sfreq']
        
        # 1) NOTCH - scipy.signal.iirnotch implementation
        if notch_freq is not None:
            b, a = iirnotch(w0=notch_freq, Q=notch_q, fs=sfreq)
            raw.notch_filter(
                freqs=notch_freq, 
                method='iir', 
                iir_params={'a': a, 'b': b}, 
                verbose=False
            )
        
        # 2) HIGH-PASS (Cutoff 1.0 Hz, ripple < 1dB at 2.0 Hz)
        if hp_cutoff is not None:
            # wp = 2.0Hz (passband), ws = 0.5Hz (stopband), gpass=1dB, gstop=20dB
            N_hp, Wn_hp = buttord(wp=2.0, ws=0.5, gpass=1, gstop=20, fs=sfreq)
            N_hp = min(N_hp, 4)  # Limit for MVAR stability
            sos_hp = butter(N_hp, Wn_hp, btype='highpass', output='sos', fs=sfreq)
            
            raw.filter(
                l_freq=hp_cutoff, 
                h_freq=None, 
                method='iir',
                iir_params={'method': 'sos', 'sos': sos_hp}, 
                phase='zero', 
                verbose=False
            )
            print(f"[Filter Info] HP Order: {N_hp}")

        # 3) LOW-PASS (40 Hz cutoff, >20dB attenuation at 50 Hz, ripple < 1dB)
        if lp_cutoff is not None:
            # wp = 40Hz, ws = 50Hz, gpass=1dB, gstop=20dB
            N_lp, Wn_lp = buttord(wp=40, ws=50, gpass=1, gstop=20, fs=sfreq)
            sos_lp = butter(N_lp, Wn_lp, btype='lowpass', output='sos', fs=sfreq)
            
            raw.filter(
                l_freq=None, 
                h_freq=lp_cutoff, 
                method='iir',
                iir_params={'method': 'sos', 'sos': sos_lp}, 
                phase='zero', 
                verbose=False
            )
            print(f"[Filter Info] LP Order: {N_lp}")

        if inplace:
            self.raw = raw
            return self
        return raw

    # ============ REFERENCE & RESAMPLE ============
    def rereference(self, ref_channels=DEFAULT_MONTAGE, verbose=False):
        self.raw.set_eeg_reference(ref_channels, verbose=verbose)
        return self

    def resample(self, sfreq=DEFAULT_SFREQ, verbose=False):
        if self.raw.info["sfreq"] != sfreq:
            self.raw.resample(float(sfreq), verbose=verbose)
        return self

    # ============ FULL PIPELINE ============
    def preprocess(
        self,
        ref_channels=DEFAULT_MONTAGE,
        sfreq=DEFAULT_SFREQ,
        notch_freq=DEFAULT_NOTCH_FREQ,
        notch_q=DEFAULT_NOTCH_Q,
        hp_cutoff=DEFAULT_HP_CUTOFF,
        lp_cutoff=DEFAULT_LP_CUTOFF,
        plot=False,
        percentile=DEFAULT_PERCENTILE
    ):
        self.standardize_channels(inplace=True)
        self.resample(sfreq)
        self.apply_filters(
            notch_freq=notch_freq,
            notch_q=notch_q,
            hp_cutoff=hp_cutoff,
            lp_cutoff=lp_cutoff,
            inplace=True,
            fs=None
        )
        self.rereference(ref_channels)

        if plot:
            scaling = {"eeg": np.percentile(np.abs(self.raw.get_data()), percentile)}
            self.raw.plot(scalings=scaling)

        return self.raw