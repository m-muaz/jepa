import os
import json
import re
import warnings
import random
import sys, logging
from logging import getLogger
from typing import List, Tuple, Optional, Dict, Any, Callable

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# Assuming utils are accessible, otherwise adjust the import path
# from src.datasets.utils.weighted_sampler import DistributedWeightedSampler # Not used here, using standard sampler

logger = getLogger(__name__)

# Regex to extract partition number from zip filename like "InternVId-FLT_4.zip"
PARTITION_REGEX = re.compile(r'.*_(\d+)\.zip')


def _extract_partition_number(zip_filename: str) -> Optional[int]:
    """Extracts the partition number from the zip filename string.

    Args:
        zip_filename: The filename string (e.g., "InternVId-FLT_4.zip").

    Returns:
        The extracted partition number as an integer, or None if not found.
    """
    match = PARTITION_REGEX.match(zip_filename)
    if match:
        return int(match.group(1))
    logger.warning(f"Could not extract partition number from zip_file: {zip_filename}")
    return None


def _find_real_video_path(
    filename: str,
    partition_num: int,
    real_video_paths: List[str]
) -> Optional[str]:
    """Finds the full path to the real video based on partition number.

    Args:
        filename: The base filename of the video.
        partition_num: The partition number the video belongs to.
        real_video_paths: List of base directories for real videos, potentially
                          containing partition identifiers.

    Returns:
        The full path to the real video file, or None if not found.
    """
    # Search for a directory containing the partition number identifier
    # Example: /path/to/real/partition_4 or /path/to/real_part_4/
    partition_identifier = f"_{partition_num}" # Adjust if naming convention differs
    target_dir = None
    for path in real_video_paths:
        if partition_identifier in os.path.basename(os.path.normpath(path)) or \
           partition_identifier in path: # Check in path string itself
            target_dir = path
            break

    if target_dir is None:
        logger.warning(f"Could not find real video directory for partition {partition_num} in {real_video_paths}")
        return None

    full_path = os.path.join(target_dir, filename)
    return full_path


# Write function to log statement for console and do a safe exit
def log_and_exit(message):
    print(message)
    sys.exit()

class InternVidDataset(Dataset):
    """
    A PyTorch Dataset for loading paired real and AI-generated videos from the InternVid dataset.

    This dataset identifies pairs of real/fake videos based on filenames and metadata,
    ensuring that the number of samples is limited by the availability of fake videos.
    It samples corresponding clips from both videos in a pair.

    Attributes:
        frames_per_clip (int): Number of frames to sample per clip.
        frame_step (int): Step size between sampled frames.
        num_clips (int): Number of clips to sample from each video.
        transform (Optional[Callable]): Transformation applied to each individual clip.
        shared_transform (Optional[Callable]): Transformation applied to the entire buffer
                                                of frames before splitting into clips.
        random_clip_sampling (bool): Whether to sample clips randomly within partitions.
        allow_clip_overlap (bool): Whether sampled clips are allowed to overlap temporally.
        filter_long_videos (int): Maximum allowed video size in bytes to load.
        video_pairs (List[Tuple[str, str]]): List of valid (real_path, fake_path) pairs.
    """
    def __init__(
        self,
        real_video_paths: List[str],
        fake_video_path: str,
        metadata_json_path: str,
        frames_per_clip: int = 8,
        frame_step: int = 4,
        num_clips: int = 1,
        transform: Optional[Callable] = None,
        shared_transform: Optional[Callable] = None,
        random_clip_sampling: bool = True,
        allow_clip_overlap: bool = False,
        filter_long_videos: int = int(10**9), # 1 GB default limit
        duration: Optional[float] = None, # duration in seconds (overrides frame_step if set)
    ):
        """
        Initializes the InternVidDataset.

        Args:
            real_video_paths: List of base directories containing real videos, potentially
                              structured by partition.
            fake_video_path: Directory containing the fake/AI-generated videos.
            metadata_json_path: Path to the JSON file containing metadata (linking fake
                                filenames to partition info via 'zip_file').
            frames_per_clip: Number of frames per sampled clip.
            frame_step: Step between frames within a clip.
            num_clips: Number of clips to extract from each video.
            transform: Optional transform applied to each clip tensor.
            shared_transform: Optional transform applied to the full frame buffer before clipping.
            random_clip_sampling: If True, sample clips randomly within video partitions.
            allow_clip_overlap: If True, clips can overlap temporally.
            filter_long_videos: Skip videos larger than this size in bytes.
            duration: Optional duration in seconds to determine frame step based on video FPS.
        """
        super().__init__()
        self.real_video_paths = real_video_paths
        self.fake_video_path = fake_video_path
        self.metadata_json_path = metadata_json_path

        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_long_videos = filter_long_videos
        self.duration = duration # Store duration

        if VideoReader is None:
            raise ImportError('Unable to import "decord", required to read videos.')

        print(f"Finding video pairs for {self.fake_video_path}")
        self.video_pairs = self._find_video_pairs()
        log_and_exit("[DEBUG] -- [EXIT]")
        if not self.video_pairs:
            raise ValueError("No valid video pairs found. Check paths and metadata.")

        logger.info(f"Initialized InternVidDataset with {len(self.video_pairs)} pairs.")

    def _find_video_pairs(self) -> List[Tuple[str, str]]:
        """Scans directories and metadata to find valid (real, fake) video pairs."""
        try:
            with open(self.metadata_json_path, 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            logger.error(f"Metadata JSON not found at: {self.metadata_json_path}")
            return []
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from: {self.metadata_json_path}")
            return []

        valid_pairs = []
        fake_files = [f for f in os.listdir(self.fake_video_path)
                      if os.path.isfile(os.path.join(self.fake_video_path, f))]

        logger.info(f"Found {len(fake_files)} potential fake videos in {self.fake_video_path}.")
        
        # Debug information
        print("First 5 fake filenames:", fake_files[:5])
        print("First 5 metadata keys:", list(metadata.keys())[:5])
        
        # Check if filenames need normalization
        sample_metadata_key = next(iter(metadata.keys()))
        print("Sample metadata key format:", sample_metadata_key)
        # Find the metadata for the first 
        log_and_exit("[DEBUG] -- [EXIT]")

        for fake_filename in fake_files:
            fake_filepath = os.path.join(self.fake_video_path, fake_filename)

            # Check fake file size first
            try:
                _fsize = os.path.getsize(fake_filepath)
                if _fsize < 1 * 1024:  # Basic check for empty/corrupt files
                    warnings.warn(f'Skipping potentially corrupt fake video (too small): {fake_filepath}')
                    continue
                if _fsize > self.filter_long_videos:
                    warnings.warn(f'Skipping long fake video (size {_fsize} > {self.filter_long_videos}): {fake_filepath}')
                    continue
            except OSError as e:
                warnings.warn(f"Could not get size for fake video {fake_filepath}: {e}")
                continue


            # --- Find corresponding real video ---
            meta_entry = metadata.get(fake_filename)
            if not meta_entry:
                warnings.warn(f"No metadata entry found for fake video: {fake_filename}")
                continue

            zip_file = meta_entry.get("zip_file")
            if not zip_file:
                warnings.warn(f"Metadata entry for {fake_filename} lacks 'zip_file' field.")
                continue

            partition_num = _extract_partition_number(zip_file)
            if partition_num is None:
                warnings.warn(f"Could not determine partition for {fake_filename} from zip_file '{zip_file}'.")
                continue

            real_filepath = _find_real_video_path(fake_filename, partition_num, self.real_video_paths)

            if real_filepath and os.path.exists(real_filepath):
                 # Check real file size
                try:
                    _rsize = os.path.getsize(real_filepath)
                    if _rsize < 1 * 1024:
                        warnings.warn(f'Skipping potentially corrupt real video (too small): {real_filepath}')
                        continue
                    if _rsize > self.filter_long_videos:
                         warnings.warn(f'Skipping long real video (size {_rsize} > {self.filter_long_videos}): {real_filepath}')
                         continue
                    valid_pairs.append((real_filepath, fake_filepath))
                except OSError as e:
                    warnings.warn(f"Could not get size for real video {real_filepath}: {e}")
                    continue
            else:
                warnings.warn(f"Real video not found for fake video {fake_filename} at expected path: {real_filepath}")

        logger.info(f"Found {len(valid_pairs)} valid (real, fake) video pairs.")
        return valid_pairs

    def _calculate_clip_indices(self, video_len: int, vr_fps: float) -> Optional[List[np.ndarray]]:
        """
        Calculates the frame indices for sampling clips, adapting logic from video_dataset.py.

        Args:
            video_len: The total number of frames in the video.
            vr_fps: The frames per second of the video reader, used if duration is set.

        Returns:
            A list of numpy arrays, where each array contains the frame indices for one clip.
            Returns None if the video is too short based on filtering settings (implicitly handled).
        """
        fpc = self.frames_per_clip
        fstp = self.frame_step

        # Calculate frame step based on duration and fps if duration is provided
        current_fstp = fstp
        if self.duration is not None:
            try:
                # Calculate total frames needed for the duration at the given step
                # Ensure fstp is at least 1
                current_fstp = max(1, int(round(self.duration * vr_fps / fpc)))
                logger.debug(f"Calculated frame_step {current_fstp} based on duration {self.duration}s and fps {vr_fps:.2f}")
            except Exception as e:
                warnings.warn(f"Could not calculate frame step from duration: {e}. Using default step {fstp}.")
                current_fstp = fstp

        clip_len_frames = int(fpc * current_fstp) # Total frames spanned by one clip

        # Basic check: Video must be long enough for at least one clip frame span
        # Note: Original video_dataset had filter_short_videos flag, here we imply it.
        if video_len < 1: # Handle empty video case
             return None
        # A single clip needs at least fpc frames, potentially spaced out.
        # The indices calculation below handles needing `clip_len_frames` total span.
        if video_len < fpc:
             warnings.warn(f"Video length {video_len} is shorter than frames_per_clip {fpc}. Cannot sample.")
             return None
        # If not overlapping, need enough frames for all clips without overlap
        if not self.allow_clip_overlap and video_len < clip_len_frames * self.num_clips:
             warnings.warn(f"Video length {video_len} too short for {self.num_clips} non-overlapping clips of span {clip_len_frames}. Skipping.")
             return None
        # If overlapping allowed, still need enough for one clip span
        if self.allow_clip_overlap and video_len < clip_len_frames:
             warnings.warn(f"Video length {video_len} too short for one clip of span {clip_len_frames}. Skipping.")
             return None


        all_clip_indices = []
        partition_len = video_len // self.num_clips # Length of each segment

        for i in range(self.num_clips):
            start_offset = i * partition_len
            end_offset = (i + 1) * partition_len

            # --- Logic adapted from video_dataset.py ---
            indices = None
            actual_clip_len_frames = min(clip_len_frames, video_len) # Max span possible

            if partition_len >= actual_clip_len_frames:
                # Sample a window within the partition
                max_start_frame = end_offset - actual_clip_len_frames
                # Ensure max_start_frame is not before the partition start
                max_start_frame = max(start_offset, max_start_frame)

                start_frame = start_offset # Default start
                if self.random_clip_sampling and max_start_frame > start_offset:
                    start_frame = np.random.randint(start_offset, max_start_frame + 1)
                elif self.random_clip_sampling: # If max_start_frame == start_offset
                     start_frame = start_offset

                end_frame = start_frame + actual_clip_len_frames
                # Generate indices within this window
                indices = np.linspace(start_frame, end_frame - 1, num=fpc, dtype=np.int64)
                # Clip indices to be within the video bounds strictly
                indices = np.clip(indices, 0, video_len - 1)

            else: # partition_len < actual_clip_len_frames
                if not self.allow_clip_overlap:
                    # Sample from the start of the partition, repeat last frame if needed
                    num_available_frames = end_offset - start_offset
                    # Calculate how many frames we can actually sample with the step
                    num_stepped_frames = max(1, num_available_frames // current_fstp)
                    sampled_indices = np.linspace(start_offset, start_offset + (num_stepped_frames - 1) * current_fstp,
                                                  num=num_stepped_frames, dtype=np.int64)

                    # Repeat the last frame if we don't have enough unique frames for fpc
                    num_repeats = fpc - len(sampled_indices)
                    if num_repeats > 0:
                        last_frame_index = sampled_indices[-1] if len(sampled_indices) > 0 else start_offset
                        # Ensure last_frame_index is valid
                        last_frame_index = min(last_frame_index, video_len - 1)
                        repeat_indices = np.full(num_repeats, last_frame_index, dtype=np.int64)
                        indices = np.concatenate((sampled_indices, repeat_indices))
                    else:
                        # If we sampled more than fpc (due to step size), take the first fpc
                        indices = sampled_indices[:fpc]

                    # Clip indices to be within the video bounds strictly
                    indices = np.clip(indices, 0, video_len - 1)


                else: # allow_clip_overlap and partition_len < actual_clip_len_frames
                    # Sample starting near the beginning of the partition, potentially overlapping
                    # Calculate the step between the *start* of each clip if overlap is allowed
                    clip_start_step = 0
                    if self.num_clips > 1 and video_len > actual_clip_len_frames:
                         # Distribute start points evenly across the possible range
                        clip_start_step = (video_len - actual_clip_len_frames) // (self.num_clips - 1)

                    start_frame = i * clip_start_step
                    end_frame = start_frame + actual_clip_len_frames
                    # Generate indices within this potentially overlapping window
                    indices = np.linspace(start_frame, end_frame - 1, num=fpc, dtype=np.int64)
                     # Clip indices to be within the video bounds strictly
                    indices = np.clip(indices, 0, video_len - 1)


            if indices is None or len(indices) != fpc:
                 warnings.warn(f"Failed to generate exactly {fpc} indices for clip {i}. Got {len(indices) if indices is not None else 'None'}. Video len: {video_len}, partition len: {partition_len}, clip span: {actual_clip_len_frames}. Skipping video pair.")
                 return None # Indicate failure

            all_clip_indices.append(indices)

        return all_clip_indices


    def _load_video_clips(
        self,
        video_path: str,
        clip_indices: List[np.ndarray]
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Loads specific clips from a video file using Decord based on pre-calculated indices.

        Args:
            video_path: Path to the video file.
            clip_indices: A list of numpy arrays, each containing frame indices for one clip.

        Returns:
            A tuple containing:
            - A numpy array of the loaded video frames (shape: [total_frames, H, W, C])
            - The average FPS of the video reader.
            Returns None if loading fails.
        """
        try:
            # Check size again just before loading (optional, mostly done in init)
            _fsize = os.path.getsize(video_path)
            if _fsize < 1 * 1024 or _fsize > self.filter_long_videos:
                 warnings.warn(f"Video {video_path} size check failed just before loading. Skipping.")
                 return None, 0.0

            vr = VideoReader(video_path, num_threads=max(1, os.cpu_count() // 2), ctx=cpu(0)) # Use half cores
            video_len = len(vr)
            if video_len == 0:
                warnings.warn(f"Video reader reported 0 length for {video_path}. Skipping.")
                return None, 0.0

            # Flatten the list of index arrays into a single list for batch loading
            all_indices = np.concatenate(clip_indices).tolist()
            # Ensure indices are within bounds (should be handled by calculation, but safety check)
            all_indices = [min(max(0, idx), video_len - 1) for idx in all_indices]

            buffer = vr.get_batch(all_indices).asnumpy()
            fps = vr.get_avg_fps()
            del vr # Release video reader resources
            return buffer, fps

        except Exception as e:
            warnings.warn(f"Failed to load video {video_path} using decord: {e}")
            return None, 0.0

    def __len__(self) -> int:
        """Returns the number of valid video pairs."""
        return len(self.video_pairs)

    def __getitem__(self, index: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Fetches the n-th pair of (real, fake) video clips.

        Args:
            index: The index of the video pair to retrieve.

        Returns:
            A tuple containing (real_clips_tensor, fake_clips_tensor), where each tensor
            has a shape like [num_clips, C, frames_per_clip, H, W] after transforms.
            Returns None if loading or processing fails for this index, expecting
            the DataLoader's collate_fn to handle it (e.g., by skipping).
        """
        real_path, fake_path = self.video_pairs[index]

        # --- Attempt to get video lengths first for consistent sampling ---
        real_len, fake_len = 0, 0
        real_fps, fake_fps = 0.0, 0.0
        try:
            # Use a temporary reader to get length and fps
            vr_real_check = VideoReader(real_path, ctx=cpu(0))
            real_len = len(vr_real_check)
            real_fps = vr_real_check.get_avg_fps()
            del vr_real_check
            vr_fake_check = VideoReader(fake_path, ctx=cpu(0))
            fake_len = len(vr_fake_check)
            fake_fps = vr_fake_check.get_avg_fps()
            del vr_fake_check
        except Exception as e:
            warnings.warn(f"Failed to open real '{real_path}' or fake '{fake_path}' to get length/fps: {e}. Skipping pair index {index}.")
            # Try next sample recursively (be careful with recursion depth)
            # A better approach is to return None and have collate_fn handle it.
            # Or pre-filter thoroughly in __init__. Let's return None.
            return None # Signal failure for this index

        # Use the shorter video length for consistent index calculation
        # Use the average FPS or prioritize real FPS if duration is used.
        reference_len = min(real_len, fake_len)
        reference_fps = real_fps if real_fps > 0 else fake_fps # Use real FPS if available

        if reference_len <= 0:
             warnings.warn(f"Reference length is zero for pair index {index} ({real_path}, {fake_path}). Skipping.")
             return None

        # --- Calculate consistent clip indices ---
        clip_indices = self._calculate_clip_indices(reference_len, reference_fps)
        if clip_indices is None:
            # Warning already issued by _calculate_clip_indices
            warnings.warn(f"Failed to calculate clip indices for pair index {index} ({real_path}, {fake_path}). Skipping.")
            return None # Signal failure

        # --- Load actual video frames using calculated indices ---
        real_buffer_tuple = self._load_video_clips(real_path, clip_indices)
        fake_buffer_tuple = self._load_video_clips(fake_path, clip_indices)

        if real_buffer_tuple is None or fake_buffer_tuple is None:
            warnings.warn(f"Failed to load buffer for real or fake video for pair index {index}. Skipping.")
            return None # Signal failure

        real_buffer, _ = real_buffer_tuple
        fake_buffer, _ = fake_buffer_tuple

        # Expected buffer shape: [total_frames_sampled, H, W, C]
        # total_frames_sampled = num_clips * frames_per_clip

        # --- Apply Transforms ---
        def process_buffer(buffer: np.ndarray) -> torch.Tensor:
            # 1. Apply shared transform (operates on the whole buffer)
            if self.shared_transform is not None:
                buffer = self.shared_transform(buffer) # Assuming it handles numpy array [T, H, W, C]

            # 2. Split buffer into clips
            # Input buffer shape: [N*F, H, W, C] -> List of [F, H, W, C]
            clips = [buffer[i * self.frames_per_clip:(i + 1) * self.frames_per_clip]
                     for i in range(self.num_clips)]

            # 3. Apply individual clip transform
            if self.transform is not None:
                # Assuming transform expects [F, H, W, C] and outputs [C, F, H, W] or similar
                clips = [self.transform(clip) for clip in clips]
            else:
                # If no transform, ensure tensor format (e.g., permute dims)
                # Example: Convert list of [F,H,W,C] numpy to tensor [N, C, F, H, W]
                 clips = [torch.from_numpy(clip).permute(3, 0, 1, 2) for clip in clips]


            # 4. Stack clips into a single tensor [N, C, F, H, W]
            clips_tensor = torch.stack(clips, dim=0)
            return clips_tensor

        try:
            real_clips_tensor = process_buffer(real_buffer)
            fake_clips_tensor = process_buffer(fake_buffer)
        except Exception as e:
            logger.error(f"Error applying transforms for pair index {index} ({real_path}, {fake_path}): {e}", exc_info=True)
            return None # Signal failure

        return real_clips_tensor, fake_clips_tensor


def make_internvid_dataloader(
    real_video_paths: List[str],
    fake_video_path: str,
    metadata_json_path: str,
    batch_size: int,
    frames_per_clip: int = 8,
    frame_step: int = 4,
    num_clips: int = 1,
    random_clip_sampling: bool = True,
    allow_clip_overlap: bool = False,
    filter_long_videos: int = int(10**9),
    duration: Optional[float] = None,
    transform: Optional[Callable] = None,
    shared_transform: Optional[Callable] = None,
    rank: int = 0,
    world_size: int = 1,
    collator: Optional[Callable] = None,
    drop_last: bool = True,
    num_workers: int = 8,
    pin_mem: bool = True,
    persistent_workers: bool = True,
    **kwargs: Any # Allow extra args
) -> Tuple[Optional[DataLoader], Optional[DistributedSampler]]:
    """
    Creates the InternVidDataset and corresponding DataLoader.

    Args:
        real_video_paths: List of base directories for real videos.
        fake_video_path: Directory containing fake videos.
        metadata_json_path: Path to the metadata JSON file.
        batch_size: Number of video pairs per batch.
        frames_per_clip: Number of frames per clip.
        frame_step: Step between frames within a clip.
        num_clips: Number of clips per video.
        random_clip_sampling: Whether to sample clips randomly.
        allow_clip_overlap: Whether clips can overlap.
        filter_long_videos: Max video size in bytes.
        duration: Optional duration in seconds to determine frame step.
        transform: Transform applied to individual clips.
        shared_transform: Transform applied to the whole frame buffer.
        rank: Process rank for distributed training.
        world_size: Total number of processes for distributed training.
        collator: Custom collate function for the DataLoader.
        drop_last: Whether to drop the last incomplete batch.
        num_workers: Number of worker processes for data loading.
        pin_mem: Whether to use pinned memory.
        persistent_workers: Whether to keep workers alive between epochs.
        **kwargs: Additional arguments (ignored).


    Returns:
        A tuple containing the DataLoader and the DistributedSampler (or None if world_size=1).
        Returns (None, None) if dataset initialization fails.
    """
    try:
        dataset = InternVidDataset(
            real_video_paths=real_video_paths,
            fake_video_path=fake_video_path,
            metadata_json_path=metadata_json_path,
            frames_per_clip=frames_per_clip,
            frame_step=frame_step,
            num_clips=num_clips,
            random_clip_sampling=random_clip_sampling,
            allow_clip_overlap=allow_clip_overlap,
            filter_long_videos=filter_long_videos,
            duration=duration,
            shared_transform=shared_transform,
            transform=transform
        )
    except ValueError as e:
        logger.error(f"Failed to initialize InternVidDataset: {e}")
        return None, None

    logger.info('InternVidDataset dataset created successfully.')

    dist_sampler = None
    shuffle = True # Shuffle by default
    if world_size > 1:
        dist_sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True, # Shuffle is recommended for training
            drop_last=drop_last # Ensure all ranks have same number of samples if True
        )
        # If using DistributedSampler, DataLoader shuffle must be False
        shuffle = False
        logger.info(f'Using DistributedSampler for InternVidDataset (rank {rank}/{world_size}).')


    # Simple collate function to handle potential None values from __getitem__
    def safe_collate(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
            # Return empty tensors or raise error if batch becomes empty
            # This depends on how the training loop handles empty batches
            logger.warning("Collate function received an empty batch after filtering Nones.")
            # Returning None might break standard training loops.
            # Return tensors with a 0 batch dimension if possible.
            # This requires knowing the expected tensor structure.
            # Example placeholder (adjust based on actual tensor structure):
            # return torch.empty(0, num_clips, C, F, H, W), torch.empty(0, num_clips, C, F, H, W)
            # For now, let the default collate handle it if the batch isn't empty.
            if collator:
                # If a custom collator is provided, it should handle Nones or empty batches.
                return collator(batch)
            # If batch is truly empty after filtering, we cannot proceed with default_collate.
        # Use default collate if a custom one isn't provided and batch is not empty
        _collator = collator if collator is not None else torch.utils.data.dataloader.default_collate
        try:
            return _collator(batch)
        except Exception as e:
            logger.error(f"Error during collate_fn: {e}", exc_info=True)
            # Return None to signal error during collation
            return None


    data_loader = DataLoader(
        dataset,
        collate_fn=safe_collate, # Use the wrapper collate
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=persistent_workers and num_workers > 0,
        shuffle=shuffle # Only True if not using DistributedSampler
    )
    logger.info('InternVidDataset data loader created.')

    # Return dataset as well if needed downstream, similar to make_videodataset?
    # Original function only returned loader and sampler. Sticking to that.
    return data_loader, dist_sampler

# Example Usage (Illustrative - requires actual data and paths)
if __name__ == '__main__':
    # Configure logging for demonstration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Dummy Data Setup (Replace with actual paths) ---
    # Create dummy directories and files for testing
    # BASE_DIR = "/blob/kyoungjun"
    
    # real_paths = [os.path.join(BASE_DIR, f"internvid_flt_{i}_reformatted") for i in range(1, 11)]
    # fake_path = os.path.join(BASE_DIR, "gen_internvid_flt")
    # metadata_file = os.path.join(BASE_DIR, "InternVid-10M-flt-zipindex.json")

    # # Create dummy video files (empty files for path existence check)
    # # In reality, these need to be actual video files loadable by decord
    # dummy_real1 = os.path.join(DUMMY_BASE, "real_part_1", "video1.mp4")
    # dummy_real2 = os.path.join(DUMMY_BASE, "real_part_2", "video2.mp4")
    # dummy_fake1 = os.path.join(DUMMY_BASE, "fake", "video1.mp4")
    # dummy_fake2 = os.path.join(DUMMY_BASE, "fake", "video2.mp4")
    # dummy_fake3_no_real = os.path.join(DUMMY_BASE, "fake", "video3_no_real.mp4") # No corresponding real
    # dummy_fake4_no_meta = os.path.join(DUMMY_BASE, "fake", "video4_no_meta.mp4") # No metadata entry
    
    # for f in [*real_paths, fake_path]:
    #     print(f)


    # --- Configuration ---
    # REAL_PATHS = [os.path.join(DUMMY_BASE, "real_part_1"), os.path.join(DUMMY_BASE, "real_part_2")]
    # FAKE_PATH = os.path.join(DUMMY_BASE, "fake")
    # META_PATH = dummy_metadata_path
    BASE_DIR = "/blob/kyoungjun"
    
    REAL_PATHS = [os.path.join(BASE_DIR, f"internvid_flt_{i}_reformatted") for i in range(1, 11)]
    FAKE_PATH = os.path.join(BASE_DIR, "gen_internvid_flt")
    META_PATH = os.path.join(BASE_DIR, "InternVid-10M-flt-zipindex.json")
    BATCH_SIZE = 1 # Keep low for testing
    FRAMES_PER_CLIP = 16
    FRAME_STEP = 4
    NUM_CLIPS = 1
    NUM_WORKERS = 0 # Set to 0 for easier debugging in main process

    # # Define dummy transforms (replace with actual torchvision/custom transforms)
    # # Example: Normalize and convert to tensor
    # from torchvision import transforms
    # dummy_clip_transform = transforms.Compose([
    #     lambda x: torch.from_numpy(x).float() / 255.0, # Assuming numpy uint8 input [F,H,W,C]
    #     lambda x: x.permute(3, 0, 1, 2) # To [C, F, H, W]
    #     # Add normalization, resizing etc. here
    # ])
    dummy_clip_transform = None

    # --- Create DataLoader ---
    # Note: This will likely fail if decord cannot load the dummy files.
    # The purpose here is to show the setup.
    try:
        dataloader, sampler = make_internvid_dataloader(
            real_video_paths=REAL_PATHS,
            fake_video_path=FAKE_PATH,
            metadata_json_path=META_PATH,
            batch_size=BATCH_SIZE,
            frames_per_clip=FRAMES_PER_CLIP,
            num_clips=NUM_CLIPS,
            num_workers=NUM_WORKERS,
            transform=dummy_clip_transform, # Apply the dummy transform
            # shared_transform=... # Add if needed
        )

        if dataloader:
            logger.info("DataLoader created. Iterating through one batch...")
            # Iterate through one batch
            try:
                for i, batch_data in enumerate(dataloader):
                    if batch_data is None:
                         logger.warning(f"Skipping batch {i} due to collation error.")
                         continue

                    real_batch, fake_batch = batch_data
                    logger.info(f"Batch {i}:")
                    logger.info(f"  Real batch shape: {real_batch.shape}") # Expected: [B, N, C, F, H, W]
                    logger.info(f"  Fake batch shape: {fake_batch.shape}") # Expected: [B, N, C, F, H, W]

                    # Add a break to only process one batch in the example
                    if i >= 0:
                        break
                logger.info("Finished iterating.")
            except ImportError:
                 logger.error("Decord or Torchvision might be missing.")
            except Exception as e:
                logger.error(f"Error during iteration (may occur if dummy files aren't valid videos): {e}", exc_info=True)
        else:
            logger.error("DataLoader creation failed.")

    finally:
        print("Cleanup phase called! (Nothing to clean up)")
        # # --- Clean up dummy data ---
        # import shutil
        # if os.path.exists(DUMMY_BASE):
        #      logger.info(f"Cleaning up dummy data directory: {DUMMY_BASE}")
        #      # shutil.rmtree(DUMMY_BASE) # Uncomment to automatically clean up
        # pass
        # # --- End Clean up ---
