# Modified from https://github.com/facebookresearch/detectron2

import os
import sys
import tempfile
import argparse
import warnings

from tqdm import tqdm

import numpy as np

import cv2

import face_alignment

from scipy.spatial import ConvexHull

import torch

from models import classifier
from utils.utils import poly2mask
from utils.augmentation import ToTensor

WINDOW_NAME = "SLF-RPM Demo"

class Predictor():
    def __init__(self, args) -> None:
        self.device = args.device
        self.transform = ToTensor()
        self._build_model(args)

    def _build_model(self, args):
        # Create SLF-RPM model
        print("\n=> Creating SLF-RPM Classifier Model: 3D ResNet-18")
        self.model = classifier.LinearClsResNet3D(model_depth=18, n_class=1)

        # Load from pretrained model
        if args.pretrained:
            if os.path.isfile(args.pretrained):
                print(f"=> Loading model weights '{args.pretrained}'")
                checkpoint = torch.load(args.pretrained, map_location="cpu")
                state_dict = checkpoint["state_dict"]
                self.model.load_state_dict(state_dict)
                print(f"=> Loaded model weights '{args.pretrained}")
            else:
                print(f"=> Error: No checkpoint found at '{args.pretrained}'")
                print("Please check your inputs agian!")
                sys.exit()

        else:
            print(
                "=> Error: Pretrained model does not specify, demo program cannot run!"
            )
            sys.exit()

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        self.model = self.model.to(args.device)
        self.model.eval()
        # print(self.model)

    def __call__(self, clip):
        assert len(clip.shape) == 4
        clip_tensor = self.transform(clip).unsqueeze(0).to(self.device)
        with torch.no_grad():              
            pred = self.model(clip_tensor)[0].item()
        return pred

class DemoProcessor():
    def __init__(self, args) -> None:
        self.clip_len = args.clip_len
        self.predictor = Predictor(args)
        
        # Detect face landmarks model
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D,
            device=args.device,
            flip_input=False,
            face_detector="blazeface",
        )
        self.ROI_forehead = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        self.ROI_cheek_left = [0, 1, 2, 3, 4, 5, 31, 41, 48]
        self.ROI_cheek_right = [16, 15, 14, 13, 12, 11, 35, 46, 54]
        self.ROI_mouth = [5, 6, 7, 8, 9, 10, 11, 54, 55, 56, 57, 58, 59, 48]

    def video_reader(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def landmarker(self, frame):
        landmark = self.fa.get_landmarks(frame)
        forehead_idx = np.zeros((1,1,2), dtype=np.uint8)
        left_cheek_idx = np.zeros((1,1,2), dtype=np.uint8)
        right_cheek_idx = np.zeros((1,1,2), dtype=np.uint8)
        mouth_idx = np.zeros((1,1,2), dtype=np.uint8)

        # Crop frame based on landmarks
        if landmark is None:
            # If landmarks cannot be detected, reture a black frame
            frame = np.zeros((64, 64), dtype=np.uint8)

        else:
            landmark = landmark[0]
            ROI_face = ConvexHull(landmark).vertices
            frame = poly2mask(
                landmark[ROI_face, 1], landmark[ROI_face, 0], frame, (64, 64)
            )
            forehead_idx = np.flip(landmark[self.ROI_forehead], -1).reshape((-1, 1, 2))
            left_cheek_idx = np.flip(landmark[self.ROI_cheek_left], -1).reshape((-1, 1, 2))
            right_cheek_idx = np.flip(landmark[self.ROI_cheek_right], -1).reshape((-1, 1, 2))
            mouth_idx = np.flip(landmark[self.ROI_mouth], -1).reshape((-1, 1, 2))

        return frame, forehead_idx, left_cheek_idx, right_cheek_idx, mouth_idx

    def run(self, video):
        width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
        fps = video.get(cv2.CAP_PROP_FPS)

        frame_loader = self.video_reader(video)  
        ori_clip = np.empty((self.clip_len, height, width, 3), dtype=np.uint8)  
        clip = np.empty((self.clip_len, 64, 64, 3), dtype=np.uint8)
        idx = 0

        for frame in frame_loader:
            if idx == self.clip_len:
                idx = 0
                pred = self.predictor(clip)
                print(f"HR: {pred:.2f}")

                for f in ori_clip:
                    vis_frame = cv2.putText(f, f"HR: {pred:.2f}", org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
                    yield vis_frame               

            else:
                ori_clip[idx] = frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame = cv2.resize(frame, (64,64))
                clip[idx], forehead_idx, left_cheek_idx, right_cheek_idx, mouth_idx = self.landmarker(frame)

                b1, g1, r1 = ori_clip[idx][forehead_idx[0][0][0], forehead_idx[0][0][1]]/255
                b2, g2, r2 = ori_clip[idx][left_cheek_idx[0][0][0], left_cheek_idx[0][0][1]]/255
                b3, g3, r3 = ori_clip[idx][right_cheek_idx[0][0][0], right_cheek_idx[0][0][1]]/255
                b4, g4, r4 = ori_clip[idx][mouth_idx[0][0][0], mouth_idx[0][0][1]]/255

                cv2.polylines(ori_clip[idx], forehead_idx, True, (0, 102, 102))
                cv2.polylines(ori_clip[idx], left_cheek_idx, True, (255, 128, 0))
                cv2.polylines(ori_clip[idx], right_cheek_idx, True, (255, 102, 102))
                cv2.polylines(ori_clip[idx], mouth_idx, True, (51, 153, 255))

                cv2.line(ori_clip[idx], (50, 50), (50-int(10*b1),50),(0, 102, 102),1)
                cv2.line(ori_clip[idx], (50, 52), (50-int(10*g1),52),(0, 102, 102),1)
                cv2.line(ori_clip[idx], (50, 54), (50-int(10*r1),54),(0, 102, 102),1)

                cv2.line(ori_clip[idx], (150, 50), (150-int(10*b2),50),(255, 128, 0),1)
                cv2.line(ori_clip[idx], (150, 52), (150-int(10*g2),52),(255, 128, 0),1)
                cv2.line(ori_clip[idx], (150, 54), (150-int(10*r2),54),(255, 128, 0),1)

                cv2.line(ori_clip[idx], (250, 50), (250-int(10*b3),50),(255, 102, 102),1)
                cv2.line(ori_clip[idx], (250, 52), (250-int(10*g3),52),(255, 102, 102),1)
                cv2.line(ori_clip[idx], (250, 54), (250-int(10*r3),54),(255, 102, 102),1)

                cv2.line(ori_clip[idx], (350, 50), (350-int(10*b4),50),(51, 153, 255),1)
                cv2.line(ori_clip[idx], (350, 52), (350-int(10*g4),52),(51, 153, 255),1)
                cv2.line(ori_clip[idx], (350, 54), (350-int(10*r4),54),(51, 153, 255),1)
                
                idx += 1

def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLF-RPM demo for builtin configs")

    parser.add_argument("--device", default="cuda:1", type=str, help="Device to run the program")
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("-i", "--input", help="Path to video file.")
    parser.add_argument(
        "-o", "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument("-l", "--clip-len", default=75, type=int)
    parser.add_argument("-p", "--pretrained", default="./mahnob_best.pth", type=str)

    args = parser.parse_args()

    print(f"=> Use device {args.device} for demo")
    processor = DemoProcessor(args)

    if args.input:
        if args.output:
            video = cv2.VideoCapture(args.input)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames_per_second = video.get(cv2.CAP_PROP_FPS)
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            basename = os.path.basename(args.input)
            codec, file_ext = (
                ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
            )
            if codec == ".mp4v":
                warnings.warn("x264 codec not available, switching to mp4v")
            if args.output:
                if os.path.isdir(args.output):
                    output_fname = os.path.join(args.output, basename)
                    output_fname = os.path.splitext(output_fname)[0] + file_ext
                else:
                    output_fname = args.output
                assert not os.path.isfile(output_fname), output_fname
                output_file = cv2.VideoWriter(
                    filename=output_fname,
                    # some installation of opencv may not support x264 (due to its license),
                    # you can try other format (e.g. MPEG)
                    fourcc=cv2.VideoWriter_fourcc(*codec),
                    fps=float(frames_per_second),
                    frameSize=(width, height),
                    isColor=True,
                )
            assert os.path.isfile(args.input)
            for vis_frame in tqdm(processor.run(video), total=num_frames):
                if args.output:
                    output_file.write(vis_frame)
                else:
                    cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                    cv2.imshow(basename, vis_frame)
                    if cv2.waitKey(1) == 27:
                        break  # esc to quit
            video.release()
            if args.output:
                output_file.release()
            else:
                cv2.destroyAllWindows()

    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"

        cam = cv2.VideoCapture(0)
        for vis in tqdm(processor.run(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()