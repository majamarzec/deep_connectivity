# Magisterka - EEG Analysis Pipeline == IN PROGRESS

Source code for Master's thesis: 
EEG signal analysis system using functional connectivity methods (especially DTF - Directed Transfer Function) and deep learning (TBA).

## Project Structure

```
magisterka/
├── src/                    # Source code package
│   ├── preprocessing/      # EEG preprocessing
│   │   ├── processing.py      # EEGPreprocessor - main preprocessing pipeline
│   │   ├── windowing.py       # EEGWindower - segmentation and artifacts
│   │   ├── frame_selector.py  # EEGFrameSelector - event-based frame selection
│   │   ├── frame_exporter.py  # EEGFrameExporter - export and visualization
│   │   ├── frame_exporter2.py # EEGFrameExporter (simple version)
│   │   ├── utils.py           # Helper functions
│   │   └── constants.py       # Constants (paths, channels, frequencies)
│   └── connectivity/       # Connectivity analysis
│       ├── mtmvar.py          # MVAR, DTF, dDTF, GPDC
│       └── DTF_EEG_reference_example.py
│
├── notebooks/              # Demonstration Jupyter notebooks
│   ├── 01_preprocessing_demo.ipynb
│   └── 02_connectivity_demo.ipynb
│
├── data/                   # Processed data
│   └── interim/
│       └── processed/
│           └── features/
│
├── results/                # Results and exports
│   └── demoDTF/           # Demo results (frames, plots, metadata)
│
├── pyproject.toml         # Package configuration
└── README.md              # This file
```

## Architecture 

The project uses a class-based approach for better organization and reusability:

### 1. **EEGPreprocessor**
Main EEG preprocessing pipeline:
- **Loading**: Loading EDF files with support for institution-specific data fixes.
- **Channel Standardization**: Automatic channel name mapping to the 10/20 system, ujednolicanie (ordering), and application of standard montages.
- **Filtering**: Application of Notch (50/60 Hz) and band-pass (Butterworth zero-phase) filters using numerically stable Second-Order Sections (SOS).
- **Re-referencing**: Support for Common Average Reference (CAR), Linked Ears, or specific channel referencing.
- **Resampling**: Adjusting the sampling frequency to optimize subsequent computational analysis (e.g., MVAR).
- **Visualization**: Integrated Power Spectral Density (PSD) plots and sensor location topography maps.



**Usage example:**
```python
from preprocessing.processing import EEGPreprocessor

# 1. Initialize and load EDF
preprocessor = EEGPreprocessor.from_edf(
    edf_path="path/to/file.edf",
    institution_id="SZC"
)

# 2. Execute full preprocessing with visualization
raw_processed = preprocessor.preprocess(
    ref_channels='average',
    sfreq=128,
    hp_cutoff=1.0,
    lp_cutoff=45.0,
    notch_freq=50.0,
    plot_psd=True,       # Displays power spectrum (proof of noise removal)
    plot_montage=True    # Displays electrode placement on the scalp
)
```

### 2. **EEGWindower**
Segmentation and artifact detection:
- Creating fixed-length frames (non-overlapping)
- Detecting flat signals (flat channels)
- Detecting high amplitudes (muscle/movement artifacts)
- Filtering frames with artifacts

**Usage example:**
```python
from preprocessing.windowing import EEGWindower

windower = EEGWindower(raw_processed)
clean_frames, info = windower.process_fixed_frames(
    frame_duration=6.0,  # seconds
    min_amplitude=1.0,   # µV
    max_amplitude=600.0  # µV
)
```

### 3. **EEGFrameSelector**
Frame selection based on annotations/events:
- Listing available annotations
- Selecting frames containing specific events
- Grouping frames by different conditions
- Group statistics analysis

**Usage example:**
```python
from preprocessing.frame_selector import EEGFrameSelector

selector = EEGFrameSelector.from_windower_results(
    raw=raw_processed,
    windower=windower,
    process_info=info
)

# Create groups
groups = selector.create_groups({
    'eyes_opened': 'Eyes_Opened',
    'eyes_closed': 'Eyes_Closed'
}, match_mode='contains')
```

### 4. **EEGFrameExporter**
Export and visualization:
- Automatic event detection in data
- Saving frames with metadata (.npz)
- Generating plots (traces, overview, timeline)
- Complete export workflow

**Usage example:**
```python
from preprocessing.frame_exporter import EEGFrameExporter

exporter = EEGFrameExporter(
    selector=selector,
    output_dir="results/my_export"
)

# Complete export
results = exporter.export_all(
    source_file="path/to/edf.edf",
    generate_plots=True
)
```

## 🔧 Installation

1. **Install package:**
```bash
cd magisterka
pip install -e .
```

2. **Dependencies are defined in `pyproject.toml`**

## Workflow

Stangard EEG analysis workflow:

1. **Preprocessing:**
   ```python
   preprocessor = EEGPreprocessor.from_edf(edf_path, institution_id)
   raw = preprocessor.preprocess()
   ```

2. **Segmentation:**
   ```python
   windower = EEGWindower(raw)
   clean_frames, info = windower.process_fixed_frames()
   ```

3. **Selection:**
   ```python
   selector = EEGFrameSelector.from_windower_results(raw, windower, info)
   groups = selector.create_groups({'eyes_opened': 'Eyes_Opened'})
   ```

4. **Export:**
   ```python
   exporter = EEGFrameExporter(selector, output_dir="results/export")
   exporter.export_all()
   ```

## Additional notes

- **Data:** ELM19 dataset is located at `/dmj/fizmed/mmarzec/licencjat_neuro/baza_elm19/`
- **Demo results:** Demonstration exports are located in `results/demoDTF/`
- **Constants:** All paths and default parameters in `src/preprocessing/constants.py`


## Development

The project uses:
- **MNE-Python** for EEG processing
- **NumPy/SciPy** for numerical computations
- **scikit-learn/torch** for ML algorithms
- **Matplotlib/Seaborn** for visualization

---

**Author:** Maja Ewa Marzec  
**Year:** 2025/6
