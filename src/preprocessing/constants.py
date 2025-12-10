import numpy as np
import os

BASE_DIR = "/dmj/fizmed/mmarzec/licencjat_neuro"
BASE_CSV_PATH = os.path.join(BASE_DIR, "baza_elm19/ELM19_info.csv")
EDF_DIR = os.path.join(BASE_DIR, "baza_elm19/ELM19_edfs")

# ============================================================================
# VALID CHANNELS
# ============================================================================

VALID_1020 = {
    'Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4',
    'T5','P3','Pz','P4','T6','O1','O2','A1','A2','EKG','Photic'
}

# ============================================================================
# CHANNEL MAPPINGS
# ============================================================================

CHNAMES_MAPPING = [
    {
        "EEG Fp1": "Fp1",
        "EEG Fp2": "Fp2",
        "EEG F7": "F7",
        "EEG F3": "F3",
        "EEG Fz": "Fz",
        "EEG F4": "F4",
        "EEG F8": "F8",
        "EEG T3": "T3",
        "EEG C3": "C3",
        "EEG Cz": "Cz",
        "EEG C4": "C4",
        "EEG T4": "T4",
        "EEG T5": "T5",
        "EEG P3": "P3",
        "EEG Pz": "Pz",
        "EEG P4": "P4",
        "EEG T6": "T6",
        "EEG O1": "O1",
        "EEG O2": "O2",
    },
    {
        "EEG Fp1": "Fp1",
        "EEG Fp2": "Fp2",
        "EEG F7": "F7",
        "EEG F3": "F3",
        "Fz_nowe": "Fz",
        "EEG F4": "F4",
        "EEG F8": "F8",
        "EEG T3": "T3",
        "EEG C3": "C3",
        "EEG Cz": "Cz",
        "EEG C4": "C4",
        "EEG T4": "T4",
        "EEG T5": "T5",
        "EEG P3": "P3",
        "EEG Pz": "Pz",
        "EEG P4": "P4",
        "EEG T6": "T6",
        "EEG O1": "O1",
        "EEG O2": "O2",
    },
    {
        "EEG FP1-REF": "Fp1",
        "EEG FP2-REF": "Fp2",
        "EEG F3-REF": "F3",
        "EEG F4-REF": "F4",
        "EEG C3-REF": "C3",
        "EEG C4-REF": "C4",
        "EEG P3-REF": "P3",
        "EEG P4-REF": "P4",
        "EEG O1-REF": "O1",
        "EEG O2-REF": "O2",
        "EEG F7-REF": "F7",
        "EEG F8-REF": "F8",
        "EEG T3-REF": "T3",
        "EEG T4-REF": "T4",
        "EEG T5-REF": "T5",
        "EEG T6-REF": "T6",
        "EEG A1-REF": "A1",
        "EEG A2-REF": "A2",
        "EEG FZ-REF": "Fz",
        "EEG CZ-REF": "Cz",
        "EEG PZ-REF": "Pz",
    },
    {
        'C3-REF': "C3",
        'C4-REF': "C4",
        'Cz-REF': "Cz",
        'EKG-REF': "EKG",
        'F3-REF': "F3",
        'F4-REF': "F4",
        'F7-REF': "F7",
        'F8-REF': "F8",
        'Fp1-REF': "Fp1",
        'Fp2-REF': "Fp2",
        'Fz-REF': "Fz",
        'O1-REF': "O1",
        'O2-REF': "O2",
        'P3-REF': "P3",
        'P4-REF': "P4",
        'Photic-REF': "Photic",
        'Pz-REF': "Pz",
        'T3-REF': "T3",
        'T4-REF': "T4",
        'T5-REF': "T5",
        'T6-REF': "T6"
    },
]

# ============================================================================
# CHANNEL NAMES
# ============================================================================

CH_NAMES = list(CHNAMES_MAPPING[0].values())
n_channels = len(CH_NAMES)

CHAN_POS = [2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24]

# ============================================================================
# FREQUENCY BANDS
# ============================================================================

FREQ_BANDS = np.array([
    [0.5, 2], [1, 3], [2, 4], [3, 6], [4, 8],
    [6, 10], [8, 13], [10, 15], [13, 18], [15, 21],
    [18, 24], [21, 27], [24, 30], [27, 40]
])

FREQ_BANDS_PH = FREQ_BANDS.copy()

FREQ_BANDS_NAMES = np.array([
    'delta','delta','delta','theta','theta',
    'alfa','alfa','beta','beta','beta',
    'beta','beta','gamma','gamma'
], dtype=str)

FREQ_BANDS_PH_BAND_NAMES = FREQ_BANDS_NAMES.copy()

# ============================================================================
# BAND LIMITS FOR POWER
# ============================================================================

BAND_LIMITS_POW = np.array([[i[0][0], i[0][1], i[1]] for i in list(zip(FREQ_BANDS, FREQ_BANDS_NAMES))])
BAND_LIMITS_POW_PH = np.array([[i[0][0], i[0][1], i[1]] for i in list(zip(FREQ_BANDS_PH, FREQ_BANDS_PH_BAND_NAMES))])

# ============================================================================
# DEFAULT PARAMETERS
# ============================================================================


DEFAULT_NOTCH_FREQ = 50.0
DEFAULT_NOTCH_Q = 5
DEFAULT_IIR_ORDER = 4
DEFAULT_HP_CUTOFF = 1.0
DEFAULT_LP_CUTOFF = 40.0

DEFAULT_SFREQ = 128
DEFAULT_PERCENTILE = 97.5
DEFAULT_MONTAGE = "average"

