import os
import csv
import sys
import time
import datetime
import traceback
import xml.dom.minidom

import face_alignment

from tqdm import tqdm

import cv2

from PIL import Image

from skimage import draw

from scipy.spatial import ConvexHull
from scipy.signal import resample

import numpy as np

import torch

EXG1_CHANNEL_IDX = 32
EXG2_CHANNEL_IDX = 33
EXG3_CHANNEL_IDX = 34


def extract_video_frame(
    video_path, frame_path=None, frame_range=(306, 2135), downsample=True
):
    """Extract frames from the given video

    Extract each frame from the given video file and store them into '.jpg' format. It
    extracts every frame of the video. If the given frame path exsits, it overwrites
    the contents if users choose that.

    Args:
            video_path (str): Required. The path of video file.

            frame_path (str): Required. The path to store extracted frames. If the path exists, it tries to
                                    remove it by asking the user.

    Raises:
            OSError: If the given video path is incorrect, or the video cannot be opened by
                            Opencv.
            ValueError: If the given specified range out of range
    """

    frames = []
    count = 0

    cap = cv2.VideoCapture()
    cap.open(video_path)
    if not cap.isOpened():
        raise OSError("Failed to open input video")

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if frame_range is not None:
        if frame_range[1] > frame_count:
            raise ValueError("Requested frame range is longer than the video")

    for frameId in range(int(frame_count)):
        ret, frame = cap.read()

        if downsample and count % 2 == 0:
            count += 1
            continue

        if frame_range is None or (
            frameId >= frame_range[0] and frameId < frame_range[1]
        ):
            if frame_path is None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                fname = "frame_{:0>4d}.png".format(frameId)
                ofname = os.path.join(frame_path, fname)
                ret = cv2.imwrite(ofname, frame)

        count += 1

    cap.release()
    return frames


def align_face(frames, fa):
    align_frames = np.empty((len(frames), 7, 64, 64, 3), dtype=np.uint8)

    for idx, frame in enumerate(frames):
        landmark = fa.get_landmarks(frame)

        # Crop frame based on landmarks
        if landmark is None:
            # If landmarks cannot be detected, reture a black frame
            frame = np.zeros((7, 64, 64, 3), dtype=np.uint8)

        else:
            # TODO: Smooth landmarks
            frame = cal_rois(frame, landmark[0], (64, 64))  # (n_roi, h, w, c)

        align_frames[idx] = frame

    return align_frames


def extract_hr_from_ecg(file_path, channel_idx, begin=5, end=35):
    import pyedflib
    import heartpy as hp
    
    signals, signals_headers, header = pyedflib.highlevel.read_edf(
        file_path, ch_nrs=channel_idx, verbose=False
    )

    sample_rate = signals_headers[0]["sample_rate"]
    start_idx = begin * sample_rate
    end_idx = end * sample_rate
    data = signals[0][start_idx:end_idx]
    filtered = hp.filter_signal(
        data, cutoff=0.05, sample_rate=sample_rate, filtertype="notch"
    )
    resampled_data = resample(filtered, len(filtered) * 2)

    # Run analysis
    wd, m = hp.process(hp.scale_data(resampled_data), sample_rate * 2)

    return m["bpm"]


def extract_mahnob_hci_dataset(dataset_path, des_path_root):
    if not os.path.exists(des_path_root):
        os.makedirs(des_path_root)

    start = time.time()
    sessions = os.listdir(dataset_path)
    sessions.sort()

    # Detect face landmarks model
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D,
        device="cuda:1",
        flip_input=False,
        face_detector="blazeface",
    )

    # Sessions
    for session in tqdm(sessions, desc="Extract MAHNOB-HCI Dataset"):
        session_path = os.path.join(dataset_path, session)

        video_file_path = None
        ecg_file_path = None
        mata_data_file_path = None

        try:
            # File paths
            for file in os.listdir(session_path):
                if file.endswith(".avi"):
                    video_file_path = os.path.join(session_path, file)
                elif file.endswith(".bdf"):
                    ecg_file_path = os.path.join(session_path, file)
                elif file.endswith(".xml"):
                    mata_data_file_path = os.path.join(session_path, file)

            if (
                video_file_path is None
                or ecg_file_path is None
                or mata_data_file_path is None
            ):
                raise OSError("Files are incomplete in {}".format(session_path))

            # Extract ground truth HR
            hr_gt = extract_hr_from_ecg(ecg_file_path, EXG2_CHANNEL_IDX)
            if np.isnan(hr_gt):
                raise ValueError("Ground truth heart rate value is NaN")

            # Extract frames
            raw_frames = extract_video_frame(video_file_path)

            # Align face
            align_frames = align_face(raw_frames, fa)

            # Get subject ID
            doc = xml.dom.minidom.parse(mata_data_file_path)
            subject_id = doc.getElementsByTagName("subject")[0].attributes["id"].value

            # Save file
            save_path = os.path.join(des_path_root, subject_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_path = os.path.join(save_path, session)
            np.savez(file_path, frames=align_frames, hr=hr_gt)
            print(
                "{} saved with frames: {}; HR: {}!\n".format(
                    file_path, align_frames.shape, hr_gt
                )
            )

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            print(e, "\n")

            # Print origin trace info
            with open("./mahnob_hci_dataset_error.txt", "a") as myfile:
                myfile.write("Session: {}\n".format(session_path))
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info, file=myfile)
                myfile.write("\n")
                del exc_info

    duration = str(datetime.timedelta(seconds=time.time() - start))
    print("It takes {} time for extracting MAHNOB-HCI dataset".format(duration))


def extract_vipl_hr_dataset(dataset_path, des_path_root):
    if not os.path.exists(des_path_root):
        os.makedirs(des_path_root)

    start = time.time()
    subjects = os.listdir(dataset_path)
    subjects.sort()

    # Detect face landmarks model
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D,
        device="cuda:1",
        flip_input=False,
        face_detector="blazeface",
    )

    # Subject
    for subject in tqdm(subjects, desc="Extract VIPL-HR-V2 Dataset"):
        subject_path = os.path.join(dataset_path, subject)

        video_file_path = []
        mata_data_file_path = None

        try:
            # File paths
            for file in os.listdir(subject_path):
                if file.endswith(".avi"):
                    video_file_path.append(os.path.join(subject_path, file))
                elif file.endswith(".csv"):
                    mata_data_file_path = os.path.join(subject_path, file)

            if len(video_file_path) != 5 or mata_data_file_path is None:
                raise OSError("Files are incomplete in {}".format(subject_path))
            video_file_path.sort()

            # Extract ground truth HR
            hr_gt = None
            fps = None
            with open(mata_data_file_path) as f:
                csv_reader = csv.reader(f, delimiter=",")
                line_count = 0
                for row in csv_reader:
                    if line_count == 1:
                        hr_gt = row[1:]
                    elif line_count == 2:
                        fps = row[1:]

                    line_count += 1

            if len(hr_gt) != 5 or len(fps) != 5:
                raise ValueError(
                    "Ground truth heart rate value or FPS value is INCORRECT!"
                )

            for idx, vid in enumerate(video_file_path):
                # Extract frames
                raw_frames = extract_video_frame(
                    vid, frame_range=None, downsample=False
                )

                # Align face
                align_frames = align_face(raw_frames, fa)

                # Save file
                if not os.path.exists(des_path_root):
                    os.makedirs(des_path_root)
                file_path = os.path.join(
                    des_path_root,
                    "{}_{}".format(vid.split("/")[-2], vid.split("/")[-1][:6]),
                )
                np.savez(file_path, frames=align_frames, hr=hr_gt[idx], fps=fps[idx])
                print(
                    "{} saved with frames: {}; HR: {}; FPS: {}!\n".format(
                        file_path, align_frames.shape, hr_gt[idx], fps[idx]
                    )
                )

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            print(e, "\n")

            # Print origin trace info
            with open("./vipl_hr_v2_dataset_error.txt", "a") as myfile:
                myfile.write("Session: {}\n".format(subject_path))
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info, file=myfile)
                myfile.write("\n")
                del exc_info

    duration = str(datetime.timedelta(seconds=time.time() - start))
    print("It takes {} time for extracting VIPL-HR-V2 dataset".format(duration))


def extract_ubfc_dataset(dataset_path, des_path_root):
    if not os.path.exists(des_path_root):
        os.makedirs(des_path_root)

    start = time.time()
    subjects = os.listdir(dataset_path)
    subjects.sort()

    # Detect face landmarks model
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D,
        device="cuda:1",
        flip_input=False,
        face_detector="blazeface",
    )

    # Subject
    for subject in tqdm(subjects, desc="Extract UBFC-rPPG Dataset"):
        subject_path = os.path.join(dataset_path, subject)

        video_file_path = None
        mata_data_file_path = None

        try:
            # File paths
            for file in os.listdir(subject_path):
                if file.endswith(".avi"):
                    video_file_path = os.path.join(subject_path, file)
                elif file.endswith(".txt"):
                    mata_data_file_path = os.path.join(subject_path, file)

            if video_file_path is None or mata_data_file_path is None:
                raise OSError("Files are incomplete in {}".format(subject_path))

            # Extract ground truth HR
            gtHR = None
            with open(mata_data_file_path, "r") as f:
                gtdata = [[float(l) for l in line.split()] for line in f.readlines()]
                gtHR = gtdata[1]

            if gtHR is None:
                raise ValueError("Ground truth heart rate value value is INCORRECT!")

            n_vid = len(gtHR) // 150

            for i in range(n_vid):
                start = i * 150
                end = (i + 1) * 150
                # Extract frames
                raw_frames = extract_video_frame(
                    video_file_path, frame_range=(start, end), downsample=False
                )

                # Align face
                align_frames = align_face(raw_frames, fa)

                # Calculate gt
                cur_gtHR = np.mean(gtHR[start:end])

                # Save file
                des_path = os.path.join(des_path_root, subject)
                if not os.path.exists(des_path):
                    os.makedirs(des_path)
                file_path = os.path.join(des_path, str(i))
                np.savez(file_path, frames=align_frames, hr=cur_gtHR)
                print(
                    "{} saved with frames: {}; HR: {}!\n".format(
                        file_path, align_frames.shape, cur_gtHR
                    )
                )

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            print(e, "\n")

            # Print origin trace info
            with open("./ubfc_rppg_dataset_error.txt", "a") as myfile:
                myfile.write("Session: {}\n".format(subject_path))
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info, file=myfile)
                myfile.write("\n")
                del exc_info

    duration = str(datetime.timedelta(seconds=time.time() - start))
    print("It takes {} time for extracting UBFC-rPPG dataset".format(duration))


def poly2mask(vertex_row_coords, vertex_col_coords, frame, crop_shape):
    h, w, c = frame.shape
    fill_row_coords, fill_col_coords = draw.polygon(
        vertex_row_coords, vertex_col_coords, (h, w)
    )
    cropped_frame = np.zeros(frame.shape, dtype=np.uint8)

    if fill_row_coords.size == 0 or fill_col_coords.size == 0:
        pass
    else:
        cropped_frame[fill_row_coords, fill_col_coords] = frame[
            fill_row_coords, fill_col_coords
        ]
        cropped_frame = cropped_frame[
            min(fill_row_coords) : max(fill_row_coords),
            min(fill_col_coords) : max(fill_col_coords),
        ]

    # Resize frame with range(0, 255) in uint8 format
    img = Image.fromarray(cropped_frame)
    cropped_frame = np.array(img.resize(crop_shape), dtype=np.uint8)

    return cropped_frame


def cal_rois(frame, landmark, crop_shape):
    ROI_face = ConvexHull(landmark).vertices
    ROI_forehead = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    ROI_cheek_left1 = [0, 1, 2, 31, 41, 0]
    ROI_cheek_left2 = [2, 3, 4, 5, 48, 31, 2]
    ROI_cheek_right1 = [16, 15, 14, 35, 46, 16]
    ROI_cheek_right2 = [14, 13, 12, 11, 54, 35, 14]
    ROI_mouth = [5, 6, 7, 8, 9, 10, 11, 54, 55, 56, 57, 58, 59, 48, 5]

    all_ROIs = np.empty((7, crop_shape[0], crop_shape[1], 3), dtype=np.uint8)

    forehead = landmark[ROI_forehead, :]
    left_eye = np.mean(landmark[36:42, :], axis=0)
    right_eye = np.mean(landmark[42:48, :], axis=0)
    eye_distance = np.linalg.norm(left_eye - right_eye)
    tmp = (
        np.mean(landmark[17:22, :], axis=0) + np.mean(landmark[22:27, :], axis=0)
    ) / 2 - (left_eye + right_eye) / 2
    tmp = eye_distance / np.linalg.norm(tmp) * 0.6 * tmp
    ROI_forehead = np.concatenate(
        [
            forehead,
            forehead[np.newaxis, -1, :] + tmp,
            forehead[np.newaxis, 0, :] + tmp,
            forehead[np.newaxis, 0, :],
        ],
        axis=0,
    )

    all_ROIs[0] = poly2mask(
        landmark[ROI_face, 1], landmark[ROI_face, 0], frame, crop_shape
    )
    all_ROIs[1] = poly2mask(ROI_forehead[:, 1], ROI_forehead[:, 0], frame, crop_shape)
    all_ROIs[2] = poly2mask(
        landmark[ROI_cheek_left1, 1], landmark[ROI_cheek_left1, 0], frame, crop_shape
    )
    all_ROIs[3] = poly2mask(
        landmark[ROI_cheek_left2, 1], landmark[ROI_cheek_left2, 0], frame, crop_shape
    )
    all_ROIs[4] = poly2mask(
        landmark[ROI_cheek_right1, 1], landmark[ROI_cheek_right1, 0], frame, crop_shape
    )
    all_ROIs[5] = poly2mask(
        landmark[ROI_cheek_right2, 1], landmark[ROI_cheek_right2, 0], frame, crop_shape
    )
    all_ROIs[6] = poly2mask(
        landmark[ROI_mouth, 1], landmark[ROI_mouth, 0], frame, crop_shape
    )

    return all_ROIs


def write_config_to_file(config, save_path):
    """Record and save current parameter settings
    Parameters
    ----------
    config : object of class `Parameters`
            Object of class `Parameters`
    save_path : str
            Path to save the file
    """

    with open(os.path.join(save_path, "config.txt"), "w") as file:
        for arg in vars(config):
            file.write(str(arg) + ": " + str(getattr(config, arg)) + "\n")


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
