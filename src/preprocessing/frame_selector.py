# src/preprocessing/frames_selector.py

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import mne
import matplotlib.pyplot as plt


class EEGFrameSelector:
    """
    Manages event-based frame selection and grouping for EEG analysis.
    Works with clean frames from EEGWindower to select frames based on annotations.
    """
    
    def __init__(
        self,
        raw: mne.io.Raw,
        clean_frames: np.ndarray,
        clean_indices: np.ndarray,
        windower: 'EEGWindower'
    ):
        """
        Initialize frame selector.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Preprocessed Raw object with annotations
        clean_frames : np.ndarray
            Clean frames from windower (n_frames, n_channels, n_samples)
        clean_indices : np.ndarray
            Original indices of clean frames
        windower : EEGWindower
            Windower object (for accessing frame times)
        """
        self.raw = raw
        self.clean_frames = clean_frames
        self.clean_indices = clean_indices
        self.windower = windower
        self.groups = {}
        
    @classmethod
    def from_windower_results(
        cls,
        raw: mne.io.Raw,
        windower: 'EEGWindower',
        process_info: Dict
    ):
        """
        Create selector from windower processing results.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Preprocessed Raw object
        windower : EEGWindower
            Windower object that processed the frames
        process_info : dict
            Info dict returned by windower.process_fixed_frames()
            
        Returns
        -------
        selector : EEGFrameSelector
        
        Example
        -------
        clean_frames, info = windower.process_fixed_frames(...)
        selector = EEGFrameSelector.from_windower_results(
            raw_preprocessed, windower, info
        )
        """
        # Get clean frames from windower
        clean_frames = windower.frames[~process_info['rejection_mask']]
        clean_indices = process_info['clean_indices']
        
        return cls(raw, clean_frames, clean_indices, windower)
    
    # ============ ANNOTATION INSPECTION ============
    
    def list_annotations(self, verbose: bool = True) -> Dict[str, int]:
        """
        List all available annotations in the data.
        
        Parameters
        ----------
        verbose : bool
            Print annotation summary
            
        Returns
        -------
        annotations : dict
            Dictionary mapping annotation names to counts
        """
        if not hasattr(self.raw, 'annotations') or len(self.raw.annotations) == 0:
            if verbose:
                print("⚠️  No annotations found in data!")
            return {}
        
        # Count annotations
        annotation_counts = {}
        for ann in self.raw.annotations:
            name = ann['description']
            annotation_counts[name] = annotation_counts.get(name, 0) + 1
        
        if verbose:
            print("="*60)
            print("AVAILABLE ANNOTATIONS")
            print("="*60)
            for i, (name, count) in enumerate(annotation_counts.items(), 1):
                # Check if point marker or interval
                durations = [ann['duration'] for ann in self.raw.annotations 
                           if ann['description'] == name]
                is_point = all(d == 0 for d in durations)
                marker_type = "point marker" if is_point else "interval"
                print(f"  {i}. {name} (n={count}, {marker_type})")
        
        return annotation_counts
    
    # ============ SINGLE EVENT SELECTION ============
    
    def select_by_event(
        self,
        event_name: Union[str, List[str]],
        match_mode: str = 'contains',
        min_overlap: float = 0.0,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Select frames containing specific event(s).
        
        Parameters
        ----------
        event_name : str or list of str
            Event name(s) to select
        match_mode : str
            'contains', 'overlap', or 'nearest'
        min_overlap : float
            Minimum overlap ratio (for 'overlap' mode)
        verbose : bool
            Print selection statistics
            
        Returns
        -------
        selected_frames : np.ndarray
            Selected frames
        selected_indices : np.ndarray
            Original indices
        info : dict
            Selection info
        """
        if verbose:
            print("="*60)
            print(f"SELECTING FRAMES BY EVENT: {event_name}")
            print("="*60)
        
        frames, indices, info = self.windower.find_frames_with_events(
            event_names=event_name,
            clean_frames=self.clean_frames,
            clean_indices=self.clean_indices,
            match_mode=match_mode,
            min_overlap=min_overlap,
            verbose=verbose
        )
        
        if len(frames) > 0 and verbose:
            print(f"\n✓ Selected {len(frames)} frames")
            print(f"  Shape: {frames.shape}")
        
        return frames, indices, info
    
    # ============ MULTI-GROUP SELECTION ============
    
    def create_groups(
        self,
        event_groups: Dict[str, Union[str, List[str]]],
        match_mode: str = 'contains',
        min_overlap: float = 0.0,
        store: bool = True,
        verbose: bool = True
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict]]:
        """
        Create multiple frame groups based on different events.
        
        Parameters
        ----------
        event_groups : dict
            Dictionary mapping group names to event name(s)
            Example: {
                'eyes_opened': 'Eyes_Opened',
                'eyes_closed': 'Eyes_Closed'
            }
        match_mode : str
            'contains', 'overlap', or 'nearest'
        min_overlap : float
            Minimum overlap ratio (for 'overlap' mode)
        store : bool
            Store groups in self.groups
        verbose : bool
            Print statistics
            
        Returns
        -------
        groups : dict
            Dictionary mapping group names to (frames, indices, info) tuples
        """
        groups = self.windower.split_frames_by_events(
            event_groups=event_groups,
            clean_frames=self.clean_frames,
            clean_indices=self.clean_indices,
            match_mode=match_mode,
            min_overlap=min_overlap,
            verbose=verbose
        )
        
        if store:
            self.groups = groups
        
        return groups
    
    # ============ GROUP ANALYSIS ============
    
    def summarize_groups(self, groups: Optional[Dict] = None, verbose: bool = True) -> Dict:
        """
        Get summary statistics for all groups.
        
        Parameters
        ----------
        groups : dict, optional
            Groups to summarize. If None, uses stored groups.
        verbose : bool
            Print summary
            
        Returns
        -------
        summary : dict
            Summary statistics for each group
        """
        if groups is None:
            groups = self.groups
        
        if not groups:
            print("⚠️  No groups available")
            return {}
        
        summary = {}
        
        for group_name, (frames, indices, info) in groups.items():
            # Calculate statistics
            mean_amplitude = np.mean(np.abs(frames))
            std_amplitude = np.std(frames)
            mean_power = np.mean(frames**2)
            max_amplitude = np.max(np.abs(frames))
            
            summary[group_name] = {
                'n_frames': len(frames),
                'shape': frames.shape,
                'indices': indices,
                'mean_amplitude': mean_amplitude,
                'std_amplitude': std_amplitude,
                'mean_power': mean_power,
                'max_amplitude': max_amplitude,
                'n_events': info['n_events']
            }
        
        if verbose:
            print("="*60)
            print("GROUP SUMMARY")
            print("="*60)
            
            for group_name, stats in summary.items():
                print(f"\nGroup: {group_name}")
                print(f"  Frames: {stats['n_frames']}")
                print(f"  Shape: {stats['shape']}")
                print(f"  Events: {stats['n_events']}")
                print(f"  Mean amplitude: {stats['mean_amplitude']:.2e} {self.windower.unit_name}")
                print(f"  Std amplitude: {stats['std_amplitude']:.2e} {self.windower.unit_name}")
                print(f"  Mean power: {stats['mean_power']:.2e} {self.windower.unit_name}²")
        
        return summary
    
    # ============ VISUALIZATION ============
    
    def plot_group_comparison(
        self,
        groups: Optional[Dict] = None,
        frame_idx: int = 0,
        show: bool = True
    ):
        """
        Plot example frame from each group for comparison.
        
        Parameters
        ----------
        groups : dict, optional
            Groups to visualize. If None, uses stored groups.
        frame_idx : int
            Which frame to plot from each group (default: first frame)
        show : bool
            Show plots immediately
        """
        if groups is None:
            groups = self.groups
        
        if not groups:
            print("⚠️  No groups available")
            return
        
        print("="*60)
        print("GROUP COMPARISON - VISUAL")
        print("="*60)
        
        for group_name, (frames, indices, info) in groups.items():
            if len(frames) == 0:
                print(f"\n⚠️  Group '{group_name}': No frames")
                continue
            
            # Get frame index (handle if frame_idx too large)
            idx = min(frame_idx, len(frames) - 1)
            original_idx = indices[idx]
            start_time, end_time = self.windower.get_frame_times(original_idx)
            
            # Create Raw fragment
            frame_raw = self.raw.copy().crop(tmin=start_time, tmax=end_time)
            
            # Calculate scaling
            frame_data = frame_raw.get_data()
            if self.windower.data_in_volts:
                scaling_value = np.percentile(np.abs(frame_data * 1e6), 99)
                scaling = {'eeg': scaling_value * 1e-6}
            else:
                scaling_value = np.percentile(np.abs(frame_data), 99)
                scaling = {'eeg': scaling_value}
            
            print(f"\nGroup: {group_name}")
            print(f"  Frame {idx}/{len(frames)-1} (original index: {original_idx})")
            print(f"  Time: {start_time:.1f} - {end_time:.1f} s")
            
            # Plot
            frame_raw.plot(
                duration=end_time - start_time,
                n_channels=len(self.raw.ch_names),
                scalings=scaling,
                title=f"Group '{group_name}' - Frame {idx} ({start_time:.1f}-{end_time:.1f}s)",
                show=show,
                block=False
            )
        
        if show:
            plt.show()
    
    def plot_frame_distribution(
        self,
        groups: Optional[Dict] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Plot distribution of frames across groups and time.
        
        Parameters
        ----------
        groups : dict, optional
            Groups to visualize. If None, uses stored groups.
        figsize : tuple
            Figure size
        """
        if groups is None:
            groups = self.groups
        
        if not groups:
            print("⚠️  No groups available")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot 1: Frame counts
        group_names = list(groups.keys())
        frame_counts = [len(groups[name][0]) for name in group_names]
        
        axes[0].bar(group_names, frame_counts, color='steelblue', alpha=0.7)
        axes[0].set_ylabel('Number of frames')
        axes[0].set_title('Frame Distribution Across Groups')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Rotate labels if many groups
        if len(group_names) > 3:
            axes[0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Timeline
        colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
        
        for i, (group_name, (frames, indices, info)) in enumerate(groups.items()):
            frame_times = []
            for idx in indices:
                start, end = self.windower.get_frame_times(idx)
                frame_times.append((start + end) / 2)  # midpoint
            
            if len(frame_times) > 0:
                axes[1].scatter(frame_times, [i] * len(frame_times), 
                              label=group_name, color=colors[i], s=50, alpha=0.6)
        
        axes[1].set_xlabel('Time (s)')
        axes[1].set_yticks(range(len(group_names)))
        axes[1].set_yticklabels(group_names)
        axes[1].set_title('Frame Timeline by Group')
        axes[1].grid(axis='x', alpha=0.3)
        axes[1].legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    # ============ EXPORT ============
    
    def get_group(self, group_name: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Get specific group data.
        
        Parameters
        ----------
        group_name : str
            Name of the group
            
        Returns
        -------
        frames : np.ndarray
            Frames in this group
        indices : np.ndarray
            Original indices
        info : dict
            Group info
        """
        if group_name not in self.groups:
            raise ValueError(f"Group '{group_name}' not found. Available: {list(self.groups.keys())}")
        
        return self.groups[group_name]
    
    def export_groups_to_dict(self) -> Dict[str, np.ndarray]:
        """
        Export all groups as simple dictionary (frames only).
        
        Returns
        -------
        export : dict
            Dictionary mapping group names to frame arrays
        """
        return {name: frames for name, (frames, _, _) in self.groups.items()}
    
    # ============ CONVENIENCE METHODS ============
    
    def quick_select(
        self,
        event_groups: Dict[str, Union[str, List[str]]],
        match_mode: str = 'contains',
        plot: bool = True,
        verbose: bool = True
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict]]:
        """
        Quick workflow: create groups, summarize, and optionally plot.
        
        Parameters
        ----------
        event_groups : dict
            Event groups to create
        match_mode : str
            Match mode for events
        plot : bool
            Show comparison plots
        verbose : bool
            Print statistics
            
        Returns
        -------
        groups : dict
            Created groups
        """
        # Create groups
        groups = self.create_groups(
            event_groups=event_groups,
            match_mode=match_mode,
            store=True,
            verbose=verbose
        )
        
        # Summarize
        if verbose:
            print()
            self.summarize_groups(groups, verbose=True)
        
        # Plot
        if plot:
            print()
            self.plot_group_comparison(groups, frame_idx=0, show=False)
            self.plot_frame_distribution(groups)
        
        return groups