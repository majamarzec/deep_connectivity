# src/preprocessing/simple_frame_exporter2.py


from pathlib import Path
from datetime import datetime
from typing import Optional, Union, Dict
import numpy as np
import mne

class EEGFrameExporter:
    """
    Simple EEG frame exporter.
    
    Features:
    - Save all frames from EEGFrameSelector
    - Save MNE plots for each frame
    - No statistics, no overview folder
    """
    
    def __init__(self, selector: 'EEGFrameSelector', output_dir: Union[str, Path] = "output"): #type: ignore
        self.selector = selector
        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / "frames"
        self.plots_dir = self.output_dir / "plots"
        
        # Create folders
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.saved_files: Dict[str, Path] = {}
        self.plot_files: Dict[str, list] = {}
    
    def save_all_frames(self, compress: bool = True):
        """Save all frames from all groups."""
        for group_name in self.selector.groups.keys():
            frames, indices, _ = self.selector.get_group(group_name)
            if len(frames) == 0:
                continue
            
            filename = f"{group_name}_frames_{self.timestamp}.npz" if compress else f"{group_name}_frames_{self.timestamp}.npy"
            filepath = self.frames_dir / filename
            
            if compress:
                np.savez_compressed(filepath, frames=frames, indices=indices)
            else:
                np.save(filepath, frames)
            
            self.saved_files[group_name] = filepath
    
    def plot_all_frames(self, show: bool = False):
        """Plot all frames using MNE and save them."""
        for group_name in self.selector.groups.keys():
            frames, indices, _ = self.selector.get_group(group_name)
            if len(frames) == 0:
                continue
            
            self.plot_files[group_name] = []
            
            for i, idx in enumerate(indices):
                start, end = self.selector.windower.get_frame_times(idx)
                frame_raw = self.selector.raw.copy().crop(tmin=start, tmax=end)
                
                fig = frame_raw.plot(show=show, block=False)
                
                plot_filename = f"{group_name}_frame_{i}_{self.timestamp}.png"
                plot_path = self.plots_dir / plot_filename
                fig.savefig(plot_path)
                fig.close()
                
                self.plot_files[group_name].append(plot_path)
    
    def export(self, compress: bool = True, show_plots: bool = False):
        """Complete export workflow: save frames + plots."""
        self.save_all_frames(compress=compress)
        self.plot_all_frames(show=show_plots)
        return {
            'saved_frames': self.saved_files,
            'plots': self.plot_files
        }