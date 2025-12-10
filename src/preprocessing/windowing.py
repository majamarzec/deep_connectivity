# src/preprocessing/windowing.py

from typing import Tuple, Dict, Optional, Union, List
import numpy as np
import mne



class EEGWindower:
    """
    Handles segmentation and artifact rejection for preprocessed EEG data.
    Designed for connectivity analysis with non-overlapping fixed frames.
    
    IMPORTANT: Handles data in Volts (V) - standard MNE unit.
    Default thresholds are automatically converted from µV to V.
    """

    def __init__(self, raw: mne.io.Raw):
        """
        Initialize windower with preprocessed MNE Raw object.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Preprocessed EEG data (expected in Volts)
        """
        self.raw = raw
        self.frames = None
        self.frame_indices = None
        self.rejected_frames = []
        self.rejection_reasons = []
        
        # Detect data units
        self._detect_units()

    def _detect_units(self):
        """Detect if data is in Volts or microVolts."""
        data = self.raw.get_data()
        typical_amplitude = np.median(np.abs(data))
        
        if typical_amplitude < 1e-3:  # < 0.001 → likely in Volts
            self.data_in_volts = True
            self.unit_name = "V"
            self.unit_scale = 1e6  # 1 µV = 1e-6 V
        else:  # likely in microVolts
            self.data_in_volts = False
            self.unit_name = "µV"
            self.unit_scale = 1.0
        
        print(f"Detected units: {self.unit_name}")
        print(f"Typical amplitude: {typical_amplitude:.2e} {self.unit_name}")

    def _to_data_units(self, value_uv: float) -> float:
        """
        Convert threshold from µV to data units (V or µV).
        
        Parameters
        ----------
        value_uv : float
            Value in microVolts (µV)
            
        Returns
        -------
        float
            Value in data units (V if data_in_volts, else µV)
        """
        if self.data_in_volts:
            return value_uv * 1e-6  # µV to V
        return value_uv

    # ============ SEGMENTATION ============
    
    def create_fixed_frames(
        self,
        frame_duration: float = 6.0,
        drop_last_incomplete: bool = True,
        store: bool = True
    ) -> np.ndarray:
        """
        Divide continuous EEG into fixed non-overlapping frames.
        
        Parameters
        ----------
        frame_duration : float
            Duration of each frame in seconds (e.g., 6 or 10)
        drop_last_incomplete : bool
            If True, drop the last frame if it's shorter than frame_duration
        store : bool
            If True, store frames and indices in instance
            
        Returns
        -------
        frames : np.ndarray
            Array of shape (n_frames, n_channels, n_samples_per_frame)
        """
        data = self.raw.get_data()
        sfreq = self.raw.info['sfreq']
        n_channels, n_samples = data.shape
        
        frame_samples = int(frame_duration * sfreq)
        n_complete_frames = n_samples // frame_samples
        
        frames = []
        indices = []
        
        # Create complete frames
        for i in range(n_complete_frames):
            start = i * frame_samples
            end = start + frame_samples
            frames.append(data[:, start:end])
            indices.append((start, end))
        
        # Handle last incomplete frame
        if not drop_last_incomplete:
            remaining_samples = n_samples % frame_samples
            if remaining_samples > 0:
                start = n_complete_frames * frame_samples
                end = n_samples
                # Pad with zeros to match frame_samples
                incomplete_frame = data[:, start:end]
                padding = np.zeros((n_channels, frame_samples - remaining_samples))
                padded_frame = np.concatenate([incomplete_frame, padding], axis=1)
                frames.append(padded_frame)
                indices.append((start, end))
        
        frames = np.array(frames)
        
        if store:
            self.frames = frames
            self.frame_indices = indices
        
        return frames

    # ============ ARTIFACT DETECTION ============
    
    def detect_flat_frames(
        self,
        frames: Optional[np.ndarray] = None,
        min_amplitude: float = 1.0,
        min_flat_channels: int = 1
    ) -> np.ndarray:
        """
        Detect frames with flat (too low amplitude) channels.
        
        Flat signal usually indicates:
        - Disconnected electrode
        - Technical malfunction
        - Saturated amplifier
        
        Parameters
        ----------
        frames : np.ndarray, optional
            Array of shape (n_frames, n_channels, n_samples).
            If None, uses stored frames.
        min_amplitude : float
            Minimum expected peak-to-peak amplitude in µV (default: 1.0)
            Will be automatically converted to V if data is in Volts
        min_flat_channels : int
            Minimum number of flat channels to reject a frame
            
        Returns
        -------
        is_flat : np.ndarray
            Boolean array indicating which frames are flat
        """
        if frames is None:
            if self.frames is None:
                raise ValueError("No frames available. Call create_fixed_frames() first.")
            frames = self.frames
        
        # Convert threshold to data units
        threshold = self._to_data_units(min_amplitude)
        
        # Calculate peak-to-peak amplitude for each channel
        peak_to_peak = np.max(frames, axis=2) - np.min(frames, axis=2)
        n_flat_per_frame = np.sum(peak_to_peak < threshold, axis=1)
        
        return n_flat_per_frame >= min_flat_channels

    def detect_high_amplitude_frames(
        self,
        frames: Optional[np.ndarray] = None,
        max_amplitude: float = 600.0,
        min_bad_channels: int = 1
    ) -> np.ndarray:
        """
        Detect frames with amplitude exceeding physiological range.
        
        High amplitude (>600 µV) typically indicates:
        - Muscle artifacts
        - Movement artifacts
        - Electrode issues
        
        Parameters
        ----------
        frames : np.ndarray, optional
            Array of shape (n_frames, n_channels, n_samples)
        max_amplitude : float
            Maximum allowed amplitude in µV (default: 600.0)
            Will be automatically converted to V if data is in Volts
        min_bad_channels : int
            Minimum number of channels exceeding threshold
            
        Returns
        -------
        has_artifact : np.ndarray
            Boolean array indicating which frames have high amplitude artifacts
        """
        if frames is None:
            if self.frames is None:
                raise ValueError("No frames available. Call create_fixed_frames() first.")
            frames = self.frames
        
        # Convert threshold to data units
        threshold = self._to_data_units(max_amplitude)
        
        # Find maximum absolute amplitude for each channel
        max_amplitudes = np.max(np.abs(frames), axis=2)
        
        # Check which channels exceed threshold
        too_high = max_amplitudes > threshold
        n_bad_per_frame = np.sum(too_high, axis=1)
        
        return n_bad_per_frame >= min_bad_channels

    # ============ REJECTION ============
    
    def reject_bad_frames(
        self,
        frames: Optional[np.ndarray] = None,
        min_amplitude: float = 1.0,
        max_amplitude: float = 600.0,
        min_flat_channels: int = 1,
        min_bad_channels: int = 1,
        verbose: bool = True
    ) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Remove frames with flat signals or high amplitude artifacts.
        
        Parameters
        ----------
        frames : np.ndarray, optional
            Array of shape (n_frames, n_channels, n_samples).
            If None, uses stored frames.
        min_amplitude : float
            Minimum expected amplitude in µV (default: 1.0)
            Automatically converted to V if needed
        max_amplitude : float
            Maximum allowed amplitude in µV (default: 600.0)
            Automatically converted to V if needed
        min_flat_channels : int
            Minimum flat channels to reject frame
        min_bad_channels : int
            Minimum bad channels to reject frame
        verbose : bool
            Print rejection statistics
            
        Returns
        -------
        clean_frames : np.ndarray
            Array of clean frames
        info : dict
            Dictionary with rejection statistics
        """
        if frames is None:
            if self.frames is None:
                raise ValueError("No frames available. Call create_fixed_frames() first.")
            frames = self.frames
        
        n_frames_total = len(frames)
        
        # Detect problematic frames
        is_flat = self.detect_flat_frames(frames, min_amplitude, min_flat_channels)
        has_high_amplitude = self.detect_high_amplitude_frames(frames, max_amplitude, min_bad_channels)
        
        # Combine rejection criteria
        to_reject = is_flat | has_high_amplitude
        
        # Store rejection info
        self.rejected_frames = np.where(to_reject)[0].tolist()
        self.rejection_reasons = []
        for i, reject in enumerate(to_reject):
            if reject:
                reasons = []
                if is_flat[i]:
                    reasons.append("flat")
                if has_high_amplitude[i]:
                    reasons.append("high_amplitude")
                self.rejection_reasons.append(reasons)
            else:
                self.rejection_reasons.append([])
        
        # Keep only clean frames
        clean_frames = frames[~to_reject]
        
        # Get indices of clean frames
        clean_indices = np.where(~to_reject)[0]
        
        # Statistics
        n_rejected = np.sum(to_reject)
        n_flat = np.sum(is_flat)
        n_high_amplitude = np.sum(has_high_amplitude)
        
        info = {
            'n_frames_total': n_frames_total,
            'n_rejected': n_rejected,
            'n_flat': n_flat,
            'n_high_amplitude': n_high_amplitude,
            'n_clean': len(clean_frames),
            'rejection_rate': n_rejected / n_frames_total if n_frames_total > 0 else 0,
            'rejection_mask': to_reject,
            'clean_indices': clean_indices
        }
        
        if verbose:
            # Show thresholds in both units
            min_amp_data = self._to_data_units(min_amplitude)
            max_amp_data = self._to_data_units(max_amplitude)
            
            print(f"Frame Rejection Summary:")
            print(f"  Data units: {self.unit_name}")
            print(f"  Thresholds: {min_amplitude} µV = {min_amp_data:.2e} {self.unit_name}, "
                  f"{max_amplitude} µV = {max_amp_data:.2e} {self.unit_name}")
            print(f"  Total frames: {n_frames_total}")
            print(f"  Rejected: {n_rejected} ({info['rejection_rate']*100:.1f}%)")
            print(f"    - Flat (< {min_amplitude} µV): {n_flat}")
            print(f"    - High amplitude (> {max_amplitude} µV): {n_high_amplitude}")
            print(f"  Clean frames: {len(clean_frames)}")
        
        return clean_frames, info

    # ============ COMPLETE PIPELINE ============
    
    def process_fixed_frames(
        self,
        frame_duration: float = 6.0,
        drop_last_incomplete: bool = True,
        min_amplitude: float = 1.0,
        max_amplitude: float = 600.0,
        min_flat_channels: int = 1,
        min_bad_channels: int = 1,
        verbose: bool = True
    ) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Complete pipeline: segment into fixed frames → detect → reject artifacts.
        
        All amplitude thresholds are in µV and automatically converted to data units.
        
        Parameters
        ----------
        frame_duration : float
            Frame duration in seconds (default: 6.0)
        drop_last_incomplete : bool
            Drop last frame if incomplete
        min_amplitude : float
            Minimum expected amplitude in µV (default: 1.0)
        max_amplitude : float
            Maximum allowed amplitude in µV (default: 600.0)
        min_flat_channels : int
            Min flat channels to reject
        min_bad_channels : int
            Min bad channels to reject
        verbose : bool
            Print statistics
            
        Returns
        -------
        clean_frames : np.ndarray
            Clean frames of shape (n_clean, n_channels, n_samples)
        info : dict
            Processing statistics with 'clean_indices' for tracking
        """
        # Create fixed frames
        self.create_fixed_frames(
            frame_duration=frame_duration,
            drop_last_incomplete=drop_last_incomplete,
            store=True
        )
        
        if verbose:
            sfreq = self.raw.info['sfreq']
            n_samples = int(frame_duration * sfreq)
            total_duration = len(self.raw) / sfreq
            print(f"Segmentation Info:")
            print(f"  Recording duration: {total_duration:.1f}s")
            print(f"  Frame duration: {frame_duration}s ({n_samples} samples @ {sfreq}Hz)")
            print(f"  Total frames: {len(self.frames)}")
            print()
        
        # Reject bad frames
        clean_frames, rejection_info = self.reject_bad_frames(
            frames=self.frames,
            min_amplitude=min_amplitude,
            max_amplitude=max_amplitude,
            min_flat_channels=min_flat_channels,
            min_bad_channels=min_bad_channels,
            verbose=verbose
        )
        
        return clean_frames, rejection_info

    # ============ UTILITIES ============
    
    def get_frame_times(self, frame_idx: int) -> Tuple[float, float]:
        """
        Get start and end time (in seconds) for a specific frame.
        
        Parameters
        ----------
        frame_idx : int
            Frame index
            
        Returns
        -------
        start_time, end_time : Tuple[float, float]
            Time in seconds
        """
        if self.frame_indices is None:
            raise ValueError("No frame indices available. Call create_fixed_frames() first.")
        
        sfreq = self.raw.info['sfreq']
        start_sample, end_sample = self.frame_indices[frame_idx]
        return start_sample / sfreq, end_sample / sfreq

    def get_rejection_report(self) -> Dict[str, any]:
        """
        Get detailed report of rejected frames.
        
        Returns
        -------
        report : dict
            Dictionary with rejection details
        """
        if not self.rejected_frames:
            return {"message": "No frames rejected"}
        
        report = {
            "n_rejected": len(self.rejected_frames),
            "rejected_indices": self.rejected_frames,
            "reasons": {}
        }
        
        # Count reasons
        flat_count = sum(1 for r in self.rejection_reasons if "flat" in r)
        high_amp_count = sum(1 for r in self.rejection_reasons if "high_amplitude" in r)
        both_count = sum(1 for r in self.rejection_reasons if len(r) == 2)
        
        report["reasons"] = {
            "flat_only": flat_count - both_count,
            "high_amplitude_only": high_amp_count - both_count,
            "both": both_count
        }
        
        return report

    # ============ EVENT-BASED SELECTION ============
    
    def find_frames_with_events(
        self,
        event_names: Union[str, List[str]],
        clean_frames: Optional[np.ndarray] = None,
        clean_indices: Optional[np.ndarray] = None,
        min_overlap: float = 0.0,
        match_mode: str = 'overlap',
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, any]]:
        """
        Find clean frames that contain or overlap with specific events.
        
        Parameters
        ----------
        event_names : str or list of str
            Event name(s) to search for (e.g., 'Eyes_Opened', 'Seizure', etc.)
            Can be a single string or list of strings
        clean_frames : np.ndarray, optional
            Clean frames array. If None, uses last processed frames.
        clean_indices : np.ndarray, optional
            Indices of clean frames in original frame list.
            If None, uses info from last processing.
        min_overlap : float, optional
            Minimum overlap ratio (0.0 to 1.0) between frame and event.
            Only used when match_mode='overlap'.
            0.0 = any overlap, 1.0 = event must span entire frame.
            Default: 0.0 (any overlap)
        match_mode : str, optional
            Mode for matching events to frames:
            - 'overlap': Event duration overlaps with frame (uses min_overlap)
            - 'contains': Frame contains the event onset (for point events with duration=0)
            - 'nearest': Assign frame to nearest event onset
            Default: 'overlap'
        verbose : bool
            Print statistics
            
        Returns
        -------
        selected_frames : np.ndarray
            Frames that contain the specified events
        selected_indices : np.ndarray
            Original indices of selected frames
        info : dict
            Selection statistics including event times and frame mapping
            
        Examples
        --------
        # Find frames containing event markers (duration=0)
        frames_eo, indices_eo, info = windower.find_frames_with_events(
            'Eyes_Opened', 
            match_mode='contains'
        )
        
        # Find frames overlapping with event intervals (duration>0)
        frames, indices, info = windower.find_frames_with_events(
            ['Seizure', 'Ictal'], 
            match_mode='overlap',
            min_overlap=0.5
        )
        
        # Find frames nearest to event markers
        frames, indices, info = windower.find_frames_with_events(
            'Stimulus',
            match_mode='nearest'
        )
        """
        # Get events from raw
        if not hasattr(self.raw, 'annotations') or len(self.raw.annotations) == 0:
            print("Warning: No annotations found in raw data.")
            return np.array([]), np.array([]), {'n_selected': 0, 'events_found': []}
        
        # Normalize event_names to list
        if isinstance(event_names, str):
            event_names = [event_names]
        
        # Use provided arrays or stored ones
        if clean_frames is None:
            if self.frames is None:
                raise ValueError("No frames available. Call process_fixed_frames() first.")
            clean_frames = self.frames
            clean_indices = np.arange(len(self.frames))
        
        if clean_indices is None:
            clean_indices = np.arange(len(clean_frames))
        
        # Get sampling frequency
        sfreq = self.raw.info['sfreq']
        
        # Find matching events
        matching_events = []
        for event_name in event_names:
            for ann in self.raw.annotations:
                if event_name.lower() in ann['description'].lower():
                    event_start = ann['onset']
                    event_duration = ann['duration']
                    event_end = event_start + event_duration
                    matching_events.append({
                        'description': ann['description'],
                        'start': event_start,
                        'end': event_end,
                        'duration': event_duration,
                        'onset': event_start
                    })
        
        if len(matching_events) == 0:
            if verbose:
                print(f"No events found matching: {event_names}")
                print(f"Available annotations: {list(set([ann['description'] for ann in self.raw.annotations]))}")
            return np.array([]), np.array([]), {'n_selected': 0, 'events_found': []}
        
        # Check if events are point markers (duration=0)
        all_zero_duration = all(event['duration'] == 0 for event in matching_events)
        if all_zero_duration and match_mode == 'overlap':
            if verbose:
                print(f"⚠️  All events have duration=0 (point markers).")
                print(f"   Automatically switching to match_mode='contains'")
            match_mode = 'contains'
        
        # Find frames based on match_mode
        selected_mask = np.zeros(len(clean_indices), dtype=bool)
        frame_event_mapping = {}
        
        if match_mode == 'contains':
            # MODE 1: Frame contains event onset (for point events)
            for i, original_idx in enumerate(clean_indices):
                frame_start, frame_end = self.get_frame_times(original_idx)
                
                for event in matching_events:
                    event_onset = event['onset']
                    
                    # Check if frame contains event onset
                    if frame_start <= event_onset < frame_end:
                        selected_mask[i] = True
                        if i not in frame_event_mapping:
                            frame_event_mapping[i] = []
                        frame_event_mapping[i].append({
                            'event': event['description'],
                            'event_onset': event_onset,
                            'frame_start': frame_start,
                            'frame_end': frame_end,
                            'time_offset': event_onset - frame_start
                        })
        
        elif match_mode == 'nearest':
            # MODE 2: Assign each frame to nearest event
            for i, original_idx in enumerate(clean_indices):
                frame_start, frame_end = self.get_frame_times(original_idx)
                frame_center = (frame_start + frame_end) / 2
                
                # Find nearest event
                nearest_event = None
                min_distance = float('inf')
                
                for event in matching_events:
                    event_time = event['onset'] if event['duration'] == 0 else (event['start'] + event['end']) / 2
                    distance = abs(frame_center - event_time)
                    
                    if distance < min_distance:
                        min_distance = distance
                        nearest_event = event
                
                if nearest_event is not None:
                    selected_mask[i] = True
                    frame_event_mapping[i] = [{
                        'event': nearest_event['description'],
                        'event_time': nearest_event['onset'],
                        'frame_center': frame_center,
                        'distance': min_distance
                    }]
        
        else:  # match_mode == 'overlap'
            # MODE 3: Overlap-based (for events with duration>0)
            for i, original_idx in enumerate(clean_indices):
                frame_start, frame_end = self.get_frame_times(original_idx)
                frame_duration = frame_end - frame_start
                
                for event in matching_events:
                    if event['duration'] == 0:
                        # Point event - check if in frame
                        if frame_start <= event['onset'] < frame_end:
                            selected_mask[i] = True
                            if i not in frame_event_mapping:
                                frame_event_mapping[i] = []
                            frame_event_mapping[i].append({
                                'event': event['description'],
                                'overlap_ratio': 1.0,  # Point is "fully" in frame
                                'event_start': event['start'],
                                'event_end': event['end']
                            })
                    else:
                        # Interval event - calculate overlap
                        overlap_start = max(frame_start, event['start'])
                        overlap_end = min(frame_end, event['end'])
                        overlap_duration = max(0, overlap_end - overlap_start)
                        overlap_ratio = overlap_duration / frame_duration
                        
                        # Check if overlap meets threshold
                        if overlap_ratio >= min_overlap:
                            selected_mask[i] = True
                            if i not in frame_event_mapping:
                                frame_event_mapping[i] = []
                            frame_event_mapping[i].append({
                                'event': event['description'],
                                'overlap_ratio': overlap_ratio,
                                'event_start': event['start'],
                                'event_end': event['end']
                            })
        
        # Select frames
        selected_frames = clean_frames[selected_mask]
        selected_indices = clean_indices[selected_mask]
        
        # Statistics
        info = {
            'n_selected': len(selected_frames),
            'n_total_clean': len(clean_frames),
            'selection_rate': len(selected_frames) / len(clean_frames) if len(clean_frames) > 0 else 0,
            'events_found': matching_events,
            'n_events': len(matching_events),
            'event_names_searched': event_names,
            'frame_event_mapping': frame_event_mapping,
            'selected_mask': selected_mask,
            'match_mode': match_mode,
            'min_overlap': min_overlap
        }
        
        if verbose:
            print(f"Event-based Frame Selection:")
            print(f"  Searched events: {event_names}")
            print(f"  Match mode: {match_mode}")
            print(f"  Events found: {len(matching_events)}")
            
            # Check if events are point markers
            zero_duration_count = sum(1 for e in matching_events if e['duration'] == 0)
            if zero_duration_count > 0:
                print(f"  Point markers (duration=0): {zero_duration_count}/{len(matching_events)}")
            
            print(f"  Total clean frames: {len(clean_frames)}")
            print(f"  Frames with events: {len(selected_frames)} ({info['selection_rate']*100:.1f}%)")
            if match_mode == 'overlap':
                print(f"  Min overlap threshold: {min_overlap*100:.0f}%")
            
            if len(matching_events) > 0:
                print(f"\n  Event details:")
                for i, event in enumerate(matching_events[:5]):  # Show first 5
                    dur_str = f"{event['duration']:.1f}s" if event['duration'] > 0 else "point marker"
                    print(f"    {i+1}. {event['description']}: {event['start']:.1f}s ({dur_str})")
                if len(matching_events) > 5:
                    print(f"    ... and {len(matching_events) - 5} more events")
        
        return selected_frames, selected_indices, info
    
    def split_frames_by_events(
        self,
        event_groups: Dict[str, Union[str, List[str]]],
        clean_frames: Optional[np.ndarray] = None,
        clean_indices: Optional[np.ndarray] = None,
        min_overlap: float = 0.0,
        match_mode: str = 'overlap',
        verbose: bool = True
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict]]:
        """
        Split clean frames into multiple groups based on different events.
        
        Parameters
        ----------
        event_groups : dict
            Dictionary mapping group names to event name(s).
            Example: {
                'eyes_opened': 'Eyes_Opened',
                'eyes_closed': 'Eyes_Closed',
                'seizure': ['Seizure', 'Ictal']
            }
        clean_frames : np.ndarray, optional
            Clean frames array
        clean_indices : np.ndarray, optional
            Indices of clean frames
        min_overlap : float
            Minimum overlap ratio (0.0 to 1.0)
        match_mode : str
            Mode for matching events to frames:
            - 'overlap': Event duration overlaps with frame
            - 'contains': Frame contains the event onset (for point events)
            - 'nearest': Assign frame to nearest event onset
            Default: 'overlap' (auto-switches to 'contains' for point markers)
        verbose : bool
            Print statistics
            
        Returns
        -------
        groups : dict
            Dictionary mapping group names to (frames, indices, info) tuples
            
        Examples
        --------
        # Split frames by condition (point markers)
        groups = windower.split_frames_by_events({
            'eyes_opened': 'Eyes_Opened',
            'eyes_closed': 'Eyes_Closed'
        }, match_mode='contains')
        
        # Split frames by intervals
        groups = windower.split_frames_by_events({
            'baseline': 'Baseline',
            'seizure': ['Seizure', 'Ictal']
        }, match_mode='overlap', min_overlap=0.5)
        """
        groups = {}
        
        if verbose:
            print(f"\n" + "="*60)
            print("SPLITTING FRAMES BY EVENTS")
            print("="*60)
        
        for group_name, event_names in event_groups.items():
            if verbose:
                print(f"\nGroup: {group_name}")
                print("-" * 40)
            
            frames, indices, info = self.find_frames_with_events(
                event_names=event_names,
                clean_frames=clean_frames,
                clean_indices=clean_indices,
                min_overlap=min_overlap,
                match_mode=match_mode,
                verbose=verbose
            )
            
            groups[group_name] = (frames, indices, info)
        
        # Summary
        if verbose:
            print(f"\n" + "="*60)
            print("SUMMARY")
            print("="*60)
            total_clean = len(clean_frames) if clean_frames is not None else len(self.frames)
            
            # Count unique frames across all groups
            all_selected_indices = set()
            for group_name, (frames, indices, _) in groups.items():
                all_selected_indices.update(indices)
                print(f"  {group_name}: {len(frames)} frames")
            
            print(f"\n  Total unique frames selected: {len(all_selected_indices)}/{total_clean}")
            
            # Check for overlaps between groups
            if len(groups) > 1:
                group_names = list(groups.keys())
                overlaps_found = False
                for i, name1 in enumerate(group_names):
                    for name2 in group_names[i+1:]:
                        indices1 = set(groups[name1][1])
                        indices2 = set(groups[name2][1])
                        overlap = indices1.intersection(indices2)
                        if len(overlap) > 0:
                            if not overlaps_found:
                                print(f"\n  ⚠️  Frame overlaps between groups:")
                                overlaps_found = True
                            print(f"    {name1} ∩ {name2}: {len(overlap)} frames")
        
        return groups