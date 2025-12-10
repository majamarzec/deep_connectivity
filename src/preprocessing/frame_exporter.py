# src/preprocessing/frame_exporter.py

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


class EEGFrameExporter:
    """
    Handles exporting and visualizing frame groups from EEGFrameSelector.
    
    Features:
    - Automatic event detection and filtering
    - Save frames with complete metadata
    - Generate visualizations (traces, heatmaps, timelines)
    - Easy data loading for analysis
    """
    
    def __init__(
        self,
        selector: 'EEGFrameSelector',
        output_dir: Union[str, Path] = "output",
        exclude_events: Optional[List[str]] = None
    ):
        """
        Initialize exporter.
        
        Parameters
        ----------
        selector : EEGFrameSelector
            Frame selector with groups
        output_dir : str or Path
            Base output directory
        exclude_events : list of str, optional
            Event names to exclude (case-insensitive)
            Default: ['Recording starts', 'Recording ends']
        """
        self.selector = selector
        self.output_dir = Path(output_dir)
        
        # Default exclude list
        if exclude_events is None:
            exclude_events = ['Recording starts', 'Recording ends']
        self.exclude_events = [e.lower() for e in exclude_events]
        
        # Create subdirectories
        self.frames_dir = self.output_dir / "frames"
        self.plots_dir = self.output_dir / "plots"
        self.metadata_dir = self.output_dir / "metadata"
        
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.saved_files = {}
    
    # ============ EVENT FILTERING ============
    
    def get_valid_events(self, verbose: bool = True) -> Dict[str, str]:
        """
        Get all events excluding those in exclude list.
        
        Parameters
        ----------
        verbose : bool
            Print filtered events
            
        Returns
        -------
        event_groups : dict
            Dictionary mapping group keys to event names
        """
        annotation_counts = self.selector.list_annotations(verbose=False)
        
        event_groups = {}
        excluded = []
        
        for event_name in annotation_counts.keys():
            # Check if in exclude list (case-insensitive)
            if any(excl in event_name.lower() for excl in self.exclude_events):
                excluded.append(event_name)
                continue
            
            # Create group key (safe filename)
            group_key = event_name.replace(' ', '_').replace('/', '_').replace('\\', '_').lower()
            event_groups[group_key] = event_name
        
        if verbose:
            print("="*60)
            print("EVENT FILTERING")
            print("="*60)
            print(f"\n✓ Valid events: {len(event_groups)}")
            for group_key, event_name in event_groups.items():
                count = annotation_counts[event_name]
                print(f"  • {event_name} → '{group_key}' (n={count})")
            
            if excluded:
                print(f"\n⚠️  Excluded events: {len(excluded)}")
                for e in excluded:
                    print(f"  • {e}")
        
        return event_groups
    
    # ============ SAVE FRAMES ============
    
    def save_frames(
        self,
        groups: Optional[Dict] = None,
        source_file: Optional[str] = None,
        compress: bool = True,
        verbose: bool = True
    ) -> Dict[str, Path]:
        """
        Save frames for each group with metadata.
        
        Parameters
        ----------
        groups : dict, optional
            Groups to save. If None, uses selector.groups
        source_file : str, optional
            Original EDF file path
        compress : bool
            Use compression (npz vs npy)
        verbose : bool
            Print progress
            
        Returns
        -------
        saved_files : dict
            Dictionary mapping group names to file paths
        """
        if groups is None:
            groups = self.selector.groups
        
        if not groups:
            print("⚠️  No groups to save")
            return {}
        
        if verbose:
            print("\n" + "="*60)
            print("SAVING FRAMES")
            print("="*60)
        
        saved_files = {}
        
        for group_name in groups.keys():
            frames, indices, group_info = self.selector.get_group(group_name)
            
            if len(frames) == 0:
                if verbose:
                    print(f"\n⚠️  Group '{group_name}': no frames - skipping")
                continue
            
            if verbose:
                print(f"\nGroup: {group_name}")
                print(f"  Frames: {len(frames)}, Shape: {frames.shape}")
            
            # Filename
            filename = f"{group_name}_frames_{self.timestamp}.npz" if compress else f"{group_name}_frames_{self.timestamp}.npy"
            filepath = self.frames_dir / filename
            
            # Prepare metadata
            metadata = {
                'frames': frames,
                'indices': indices,
                'n_frames': len(frames),
                'n_events': group_info['n_events'],
                'event_names': group_info['event_names_searched'],
                'match_mode': group_info['match_mode'],
                'channel_names': self.selector.raw.ch_names,
                'sfreq': float(self.selector.raw.info['sfreq']),
                'frame_duration': float((frames.shape[2] / self.selector.raw.info['sfreq'])),
                'timestamp': self.timestamp,
                'data_units': self.selector.windower.unit_name
            }
            
            if source_file:
                metadata['original_file'] = str(source_file)
            
            # Save
            if compress:
                np.savez_compressed(filepath, **metadata)
            else:
                np.save(filepath, metadata)
            
            saved_files[group_name] = filepath
            
            if verbose:
                size_kb = filepath.stat().st_size / 1024
                print(f"  ✓ Saved: {filepath.name} ({size_kb:.1f} KB)")
        
        self.saved_files = saved_files
        
        if verbose:
            print(f"\n✓ Saved {len(saved_files)} groups to {self.frames_dir}")
        
        return saved_files
    
    # ============ SAVE METADATA ============
    
    def save_session_metadata(
        self,
        event_groups: Dict[str, str],
        annotation_counts: Dict[str, int],
        processing_info: Dict,
        summary: Dict,
        verbose: bool = True
    ):
        """
        Save session metadata.
        
        Parameters
        ----------
        event_groups : dict
            Event groups mapping
        annotation_counts : dict
            Annotation counts
        processing_info : dict
            Processing info from windower
        summary : dict
            Group summary statistics
        verbose : bool
            Print progress
        """
        metadata_file = self.metadata_dir / f"session_metadata_{self.timestamp}.npz"
        
        np.savez(
            metadata_file,
            event_groups=event_groups,
            annotation_counts=annotation_counts,
            processing_info=processing_info,
            summary=summary,
            saved_files=self.saved_files,
            timestamp=self.timestamp
        )
        
        if verbose:
            print(f"✓ Session metadata saved: {metadata_file.name}")
    
    # ============ VISUALIZATION ============
    
    def plot_example_trace(
        self,
        group_name: str,
        frame_idx: int = 0,
        figsize: Tuple[int, int] = (14, 12),
        dpi: int = 150,
        save: bool = True,
        show: bool = False
    ) -> Optional[Path]:
        """
        Plot example EEG trace for a group.
        
        Parameters
        ----------
        group_name : str
            Name of the group
        frame_idx : int
            Which frame to plot
        figsize : tuple
            Figure size
        dpi : int
            DPI for saved image
        save : bool
            Save to file
        show : bool
            Display plot
            
        Returns
        -------
        filepath : Path or None
            Path to saved file if save=True
        """
        frames, indices, group_info = self.selector.get_group(group_name)
        
        if len(frames) == 0:
            print(f"⚠️  No frames in group '{group_name}'")
            return None
        
        # Get frame
        idx = min(frame_idx, len(frames) - 1)
        original_idx = indices[idx]
        start_time, end_time = self.selector.windower.get_frame_times(original_idx)
        
        # Get data
        frame_raw = self.selector.raw.copy().crop(tmin=start_time, tmax=end_time)
        frame_data = frame_raw.get_data()
        
        # Convert units for plotting
        if self.selector.windower.data_in_volts:
            data_plot = frame_data * 1e6  # V → µV
            unit = 'µV'
        else:
            data_plot = frame_data
            unit = 'µV'
        
        scaling_value = np.percentile(np.abs(data_plot), 99)
        
        # Plot
        fig, axes = plt.subplots(
            len(self.selector.raw.ch_names), 1,
            figsize=figsize,
            sharex=True
        )
        
        times = np.arange(data_plot.shape[1]) / self.selector.raw.info['sfreq']
        
        for ch_idx, ch_name in enumerate(self.selector.raw.ch_names):
            axes[ch_idx].plot(times, data_plot[ch_idx, :], linewidth=0.5, color='black')
            axes[ch_idx].set_ylabel(ch_name, fontsize=8)
            axes[ch_idx].set_ylim(-scaling_value, scaling_value)
            axes[ch_idx].grid(alpha=0.3)
            axes[ch_idx].tick_params(labelsize=7)
        
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(
            f"Group: {group_name} - Frame {idx}/{len(frames)-1} ({start_time:.1f}-{end_time:.1f}s)\n"
            f"Total frames: {len(frames)} | Events: {group_info['n_events']}",
            fontsize=12, fontweight='bold'
        )
        
        plt.tight_layout()
        
        filepath = None
        if save:
            filepath = self.plots_dir / f"{group_name}_example_trace.png"
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return filepath
    
    def plot_overview(
        self,
        group_name: str,
        figsize: Tuple[int, int] = (14, 10),
        dpi: int = 150,
        save: bool = True,
        show: bool = False
    ) -> Optional[Path]:
        """
        Plot overview heatmap for a group.
        
        Parameters
        ----------
        group_name : str
            Name of the group
        figsize : tuple
            Figure size
        dpi : int
            DPI for saved image
        save : bool
            Save to file
        show : bool
            Display plot
            
        Returns
        -------
        filepath : Path or None
            Path to saved file if save=True
        """
        frames, indices, group_info = self.selector.get_group(group_name)
        
        if len(frames) == 0:
            print(f"⚠️  No frames in group '{group_name}'")
            return None
        
        # Calculate mean amplitudes
        mean_amps = np.mean(np.abs(frames), axis=2)  # (n_frames, n_channels)
        
        unit = self.selector.windower.unit_name
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Heatmap
        im = axes[0].imshow(
            mean_amps.T,
            aspect='auto',
            cmap='viridis',
            interpolation='nearest'
        )
        axes[0].set_ylabel('Channel')
        axes[0].set_xlabel('Frame Index')
        axes[0].set_title(f'Mean Amplitude per Frame - {group_name}')
        axes[0].set_yticks(range(len(self.selector.raw.ch_names)))
        axes[0].set_yticklabels(self.selector.raw.ch_names, fontsize=7)
        plt.colorbar(im, ax=axes[0], label=f'Amplitude ({unit})')
        
        # Global mean
        global_mean = np.mean(mean_amps, axis=1)
        axes[1].plot(global_mean, linewidth=1.5, marker='o', markersize=3)
        axes[1].set_xlabel('Frame Index')
        axes[1].set_ylabel(f'Mean Amplitude ({unit})')
        axes[1].set_title(f'Global Mean Amplitude per Frame - {group_name}')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        filepath = None
        if save:
            filepath = self.plots_dir / f"{group_name}_overview.png"
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return filepath
    
    def plot_timeline(
        self,
        group_name: str,
        figsize: Tuple[int, int] = (14, 4),
        dpi: int = 150,
        save: bool = True,
        show: bool = False
    ) -> Optional[Path]:
        """
        Plot timeline with frames and events.
        
        Parameters
        ----------
        group_name : str
            Name of the group
        figsize : tuple
            Figure size
        dpi : int
            DPI for saved image
        save : bool
            Save to file
        show : bool
            Display plot
            
        Returns
        -------
        filepath : Path or None
            Path to saved file if save=True
        """
        frames, indices, group_info = self.selector.get_group(group_name)
        
        if len(frames) == 0:
            print(f"⚠️  No frames in group '{group_name}'")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Frame positions
        frame_times = []
        for idx in indices:
            start, end = self.selector.windower.get_frame_times(idx)
            frame_times.append((start + end) / 2)
        
        ax.scatter(frame_times, [1]*len(frame_times), s=100, alpha=0.6, label='Frames')
        
        # Events
        for event in group_info['events_found']:
            ax.axvline(event['onset'], color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        ax.set_xlabel('Time (s)')
        ax.set_yticks([])
        ax.set_title(f'Timeline - {group_name} ({len(frames)} frames, {group_info["n_events"]} events)')
        ax.set_ylim(0.5, 1.5)
        ax.grid(axis='x', alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        filepath = None
        if save:
            filepath = self.plots_dir / f"{group_name}_timeline.png"
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return filepath
    
    def visualize_all_groups(
        self,
        groups: Optional[Dict] = None,
        dpi: int = 150,
        show: bool = False,
        verbose: bool = True
    ) -> Dict[str, Dict[str, Path]]:
        """
        Generate all visualizations for all groups.
        
        Parameters
        ----------
        groups : dict, optional
            Groups to visualize. If None, uses selector.groups
        dpi : int
            DPI for saved images
        show : bool
            Display plots
        verbose : bool
            Print progress
            
        Returns
        -------
        plot_files : dict
            Dictionary mapping group names to plot file paths
        """
        if groups is None:
            groups = self.selector.groups
        
        if not groups:
            print("⚠️  No groups to visualize")
            return {}
        
        if verbose:
            print("\n" + "="*60)
            print("GENERATING VISUALIZATIONS")
            print("="*60)
        
        plot_files = {}
        
        for group_name in groups.keys():
            frames, _, _ = self.selector.get_group(group_name)
            
            if len(frames) == 0:
                continue
            
            if verbose:
                print(f"\nGroup: {group_name}")
            
            plots = {}
            
            # Trace
            p1 = self.plot_example_trace(group_name, dpi=dpi, save=True, show=show)
            if p1:
                plots['trace'] = p1
                if verbose:
                    print(f"  ✓ Trace: {p1.name}")
            
            # Overview
            p2 = self.plot_overview(group_name, dpi=dpi, save=True, show=show)
            if p2:
                plots['overview'] = p2
                if verbose:
                    print(f"  ✓ Overview: {p2.name}")
            
            # Timeline
            p3 = self.plot_timeline(group_name, dpi=dpi, save=True, show=show)
            if p3:
                plots['timeline'] = p3
                if verbose:
                    print(f"  ✓ Timeline: {p3.name}")
            
            plot_files[group_name] = plots
        
        if verbose:
            print(f"\n✓ Generated visualizations for {len(plot_files)} groups")
        
        return plot_files
    
    # ============ COMPLETE WORKFLOW ============
    
    def export_all(
        self,
        source_file: Optional[str] = None,
        match_mode: str = 'contains',
        compress: bool = True,
        generate_plots: bool = True,
        plot_dpi: int = 150,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Complete export workflow: detect events → create groups → save → visualize.
        
        Parameters
        ----------
        source_file : str, optional
            Original EDF file path
        match_mode : str
            Match mode for events ('contains', 'overlap', 'nearest')
        compress : bool
            Use compression for saved files
        generate_plots : bool
            Generate visualization plots
        plot_dpi : int
            DPI for saved plots
        verbose : bool
            Print progress
            
        Returns
        -------
        results : dict
            Dictionary with saved files and plot files
        """
        if verbose:
            print("\n" + "="*60)
            print("EXPORT WORKFLOW")
            print("="*60)
            print(f"Output directory: {self.output_dir}")
            print(f"Timestamp: {self.timestamp}")
        
        # 1. Get valid events
        event_groups = self.get_valid_events(verbose=verbose)
        
        if not event_groups:
            print("\n⚠️  No valid events found - aborting export")
            return {'error': 'No valid events'}
        
        # 2. Create groups
        if verbose:
            print("\n" + "="*60)
            print("CREATING GROUPS")
            print("="*60)
        
        groups = self.selector.create_groups(
            event_groups=event_groups,
            match_mode=match_mode,
            store=True,
            verbose=verbose
        )
        
        # 3. Get summary
        summary = self.selector.summarize_groups(verbose=verbose)
        
        # 4. Save frames
        saved_files = self.save_frames(
            groups=groups,
            source_file=source_file,
            compress=compress,
            verbose=verbose
        )
        
        # 5. Save metadata
        annotation_counts = self.selector.list_annotations(verbose=False)
        processing_info = {
            'n_frames_total': len(self.selector.windower.frames) if self.selector.windower.frames is not None else 0,
            'n_clean': len(self.selector.clean_frames),
            'timestamp': self.timestamp
        }
        
        self.save_session_metadata(
            event_groups=event_groups,
            annotation_counts=annotation_counts,
            processing_info=processing_info,
            summary=summary,
            verbose=verbose
        )
        
        # 6. Generate plots
        plot_files = {}
        if generate_plots:
            plot_files = self.visualize_all_groups(
                groups=groups,
                dpi=plot_dpi,
                show=False,
                verbose=verbose
            )
        
        # Summary
        if verbose:
            print("\n" + "="*60)
            print("EXPORT COMPLETE")
            print("="*60)
            print(f"✓ Saved {len(saved_files)} frame groups")
            print(f"✓ Generated {sum(len(p) for p in plot_files.values())} plots")
            print(f"✓ Output: {self.output_dir}")
        
        return {
            'saved_files': saved_files,
            'plot_files': plot_files,
            'groups': groups,
            'summary': summary,
            'timestamp': self.timestamp
        }
    
    # ============ UTILITY ============
    
    @staticmethod
    def load_frames(filepath: Union[str, Path]) -> Dict:
        """
        Load saved frames file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to .npz file
            
        Returns
        -------
        data : dict
            Loaded data with frames and metadata
        """
        data = np.load(filepath, allow_pickle=True)
        return {key: data[key] for key in data.files}
    
    def print_summary(self):
        """Print summary of export."""
        print("\n" + "="*60)
        print("EXPORT SUMMARY")
        print("="*60)
        print(f"\n📁 Output directory: {self.output_dir}")
        
        if self.saved_files:
            print(f"\n📦 Saved frames:")
            for name, path in self.saved_files.items():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"   • {name}: {path.name} ({size_mb:.2f} MB)")
        
        plot_files = list(self.plots_dir.glob("*.png"))
        if plot_files:
            print(f"\n📊 Plots: {len(plot_files)} files")
        
        metadata_files = list(self.metadata_dir.glob("*.npz"))
        if metadata_files:
            print(f"\n📋 Metadata: {len(metadata_files)} files")