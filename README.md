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

## 🧠 EEG Data Preprocessing Pipeline

`EEGPreprocessor` is a robust, MNE-based tool designed for standardizing and cleaning EEG data across different medical institutions. It ensures that raw EDF files are transformed into a clean, uniform format ready for advanced analysis like **MVAR (Multivariate Autoregressive Models)**.

### ✨ Key Features
* **Smart Loading**: Specialized EDF loader with institution-specific fixes (e.g., handling data artifacts from different hospitals).
* **Channel Standardization**: 
    * Automatic mapping of non-standard clinical names to the **International 10-20 System**.
    * Automatic channel reordering for consistency across datasets.
    * Standard montage application (`standard_1020`).
* **Precision Filtering**:
    * **Notch filter** (50/60 Hz) to remove power line noise.
    * **Butterworth zero-phase filters** (High-pass & Low-pass) using `scipy.signal.sosfiltfilt` for maximum numerical stability.
    * Automated filter order selection using `buttord` to meet specific passband/stopband requirements.
* **Advanced Visualization**: 
    * **PSD (Power Spectral Density)** plots to verify noise removal.
    * **Sensor maps** to confirm correct electrode placement.
    * Signal browsers for manual data inspection.

### 🚀 Usage Example

```python
from preprocessing.processing import EEGPreprocessor

# 1. Initialize from EDF
preprocessor = EEGPreprocessor.from_edf(
    edf_path="data/recording.edf",
    institution_id="SZC"
)

# 2. Run the full pipeline
raw_clean = preprocessor.preprocess(
    sfreq=128,              # Resample to 128Hz
    ref_channels='average', # Common Average Reference (CAR)
    hp_cutoff=1.0,          # High-pass to remove DC drift
    lp_cutoff=45.0,         # Low-pass to remove high-frequency noise
    notch_freq=50.0,        # Polish power grid frequency
    plot_psd=True,          # Show frequency spectrum
    plot_montage=True       # Show electrode locations
)

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
