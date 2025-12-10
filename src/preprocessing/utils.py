from typing import List
import mne

def chunk(to_chunk: list, step: int):
    return [to_chunk[i:i + step] for i in range(0, len(to_chunk), step)]

def apply_mor_data_hack_fix(edf: mne.io.Raw, edf_path: str, institution_id: str):
    if institution_id == 'MOR':
        mor_bad_channel = 'Fz_nowe'
        if mor_bad_channel in edf.ch_names:
            if edf._orig_units[mor_bad_channel] == 'n/a':
                fixed_edf = mne.io.read_raw_edf(edf_path, preload=False)                
                bad_channel_id = edf.ch_names.index(mor_bad_channel)
                fixed_edf._raw_extras[0]['units'][bad_channel_id] = 1.e-06
                fixed_edf._orig_units[mor_bad_channel] = 'µV'
                fixed_edf.load_data()
                return fixed_edf
    return edf
