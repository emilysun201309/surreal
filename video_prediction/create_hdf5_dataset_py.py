import argparse
import os
import tempfile
from subprocess import check_call, DEVNULL, STDOUT

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
KEYPOINTS_PATH = "keypoints.h5"

def read_chest_location(file_name,store_keypoints):
    key_store = pd.HDFStore(store_keypoints)
    keypoints = key_store[file_name].iloc[:,2:4]
    keypoints = (keypoints.values).astype(int)
    return keypoints


def extract_video_frames(video_path, save_root):
    """Saves video frames as PNGs under the folder at save_root.

    :param video_path: The path to the video file
    :param save_root: The path to the folder under which all frames will be saved
    """
    cmd = 'ffmpeg -i {} -vsync 0 {}/frame_%08d.png'.format(video_path, save_root)
    check_call(cmd.split(), stdout=DEVNULL, stderr=STDOUT)


def save_video_to_hdf5(h5_file_handle, video_path):
    """Saves the frames of the specified video as a dataset in the given HDF5 file.

    :param h5_file_handle: The handle to an HDF5 file
    :param video_path: The path to the video whose frames should be added to the HDF5 file
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save video frames to disk
        # TODO: Properly handle case when frame extraction fails
        ret_code = extract_video_frames(video_path, tmp_dir)
        frame_file_names = sorted(os.listdir(tmp_dir))

        # Get size of the final video tensor
        T = len(frame_file_names)
        first_image = Image.open(os.path.join(tmp_dir, frame_file_names[0]))
        first_image_np = np.array(first_image)
        H, W, C = first_image_np.shape

        H,W = 64,64
        # Store the frames in the HDF5 file
        video_dataset = h5_file_handle.create_dataset(os.path.basename(video_path.split('.')[0]), shape=(T, H, W, C), dtype=np.uint8,
                                                      chunks=(1, H, W, C), compression='gzip')
        tqdm.write('Saving video frames to HDF5 file...')
        frame_pbar = tqdm(total=len(frame_file_names), desc='Frame (cur. video)', unit='frames')
        centers = read_chest_location(os.path.basename(video_path.split('.')[0]),KEYPOINTS_PATH)
        print(centers.shape)
        print(len(frame_file_names))
        for t, file in enumerate(frame_file_names[:-2]):
            w,h = centers[t]
            
            h = max(90,h)
            h = min(150,h)
            w = max(90,w)
            w = min(230,w)
            
            image = Image.open(os.path.join(tmp_dir, file))
            image_arr = np.array(image)
            image_arr = image_arr[h-90:h+90,w-90:w+90]

            im = Image.fromarray(image_arr)
            im = im.resize((64, 64))

            video_dataset[t, :, :, :] = np.array(im)
            frame_pbar.update()
        frame_pbar.close()


def save_videos_to_hdf5(video_paths, save_path):
    """Creates an HDF5 file at save_path and save the frames of the given videos in the file.

    :param video_paths: list of video paths to store frames of
    :param save_path: The path to the HDF5 file to create
    """
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    h5_file_handle = h5py.File(save_path, 'w')

    video_pbar = tqdm(total=len(video_paths), desc='Videos', unit='video')
    for i, video_path in enumerate(video_paths):
        tqdm.write('Extracting video frames from {} (video {})'.format(os.path.basename(video_path), i))
        save_video_to_hdf5(h5_file_handle, video_path)
        video_pbar.update()
    video_pbar.close()

    h5_file_handle.close()

    tqdm.write('\nDone.')


def save_video_list_to_hdf5(video_list_path, save_path):
    """Store the unique videos in the given video list in an HDF5 file.

    :param video_list_path: Path to a video list
    :param save_path: The path to the HDF5 file to create
    """
    with open(video_list_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    # Remove frame indexes if found
    lines = [line.split()[0] for line in lines]

    # Get unique lines
    unique_video_paths = np.unique(lines)
    # Check that the basenames are also unique
    unique_basenames = np.unique([os.path.basename(path) for path in unique_video_paths])
    if len(unique_video_paths) != len(unique_basenames):
        raise RuntimeError('At least one duplicate video name was found')

    save_videos_to_hdf5(unique_video_paths, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('video_list', type=str, help='Path to the video list')
    parser.add_argument('hdf5_path', type=str, help='Path to the dataset to create')
    args = parser.parse_args()

    save_video_list_to_hdf5(args.video_list, args.hdf5_path)