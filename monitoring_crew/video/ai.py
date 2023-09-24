import yolov7
import cv2
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from tqdm import tqdm
from PIL import Image

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import non_max_suppression_kpt, xywh2xyxy
from yolov7.utils.plots import output_to_keypoint, plot_one_box, plot_skeleton_kpts

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_pose_prediction(img, pred, thickness=1, show_bbox=True):
    bbox = xywh2xyxy(pred[:, 2:6])
    for idx in range(pred.shape[0]):
        plot_skeleton_kpts(img, pred[idx, 7:].T, 3)
        if show_bbox:
            plot_one_box(bbox[idx], img, line_thickness=thickness)


def scale_pose_output(output, resized_shape, original_shape, is_padded=True):
    scaled_output = output.copy()
    if len(scaled_output) > 0:
        scale_ratio = resized_shape[1] / original_shape[1], resized_shape[0] / original_shape[0]
        if is_padded:
            pad_scale = min(scale_ratio)
            padding = (resized_shape[1] - original_shape[1] * pad_scale) / 2, (
                    resized_shape[0] - original_shape[0] * pad_scale) / 2
            scale_ratio = (pad_scale, pad_scale)

            scaled_output[:, 2] -= padding[0]
            scaled_output[:, 3] -= padding[1]
            scaled_output[:, 7::3] -= padding[0]
            scaled_output[:, 8::3] -= padding[1]

            scaled_output[:, [2, 4]] /= scale_ratio[0]
            scaled_output[:, [3, 5]] /= scale_ratio[1]
            scaled_output[:, 7::3] /= scale_ratio[0]
            scaled_output[:, 8::3] /= scale_ratio[1]

    return scaled_output


def make_pose_prediction(img, model, frame_width):
    img_ = letterbox(img, frame_width, stride=64, auto=True)[0]
    resized_shape = img_.shape[0:2]
    img_ = transforms.ToTensor()(img_)
    img_ = torch.tensor(np.array([img_.numpy()]))
    img_ = img_.to(device).float()
    with torch.no_grad():
        output, _ = model(img_)
    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
    output = output_to_keypoint(output)
    output = scale_pose_output(output, resized_shape, img.shape[0:2])
    return output


def detect_video(path):
    POSE_WEIGHTS = '/Users/kirill201/Desktop/Projects/Monitoring_locomotive_crew/monitoring_crew/best.pt'

    ...
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = attempt_load(POSE_WEIGHTS, map_location=device)

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened():
        print('Error while trying to read video. Please check path again')

    # Get the frame width and height.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define codec and create VideoWriter object .
    out = cv2.VideoWriter('"result_mp4"',
                          cv2.VideoWriter_fourcc(*'mp4v'), fps,
                          (frame_width, frame_height))

    while cap.isOpened:
        # Capture each frame of the video.
        ret, frame = cap.read()
        if ret:
            orig_image = frame
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

            output = make_pose_prediction(model=model, img=image, frame_width=frame_width)
            # output = get_back_keypoints(pred=output, frame=orig_image, img=image_resized.shape)
            for idx in range(output.shape[0]):
                plot_skeleton_kpts(orig_image, output[idx, 7:].T, 3)

                # Show bounding boxes around persons.
                xmin, ymin = (output[idx, 2] - output[idx, 4] / 2), (output[idx, 3] - output[idx, 5] / 2)
                xmax, ymax = (output[idx, 2] + output[idx, 4] / 2), (output[idx, 3] + output[idx, 5] / 2)
                cv2.rectangle(
                    orig_image,
                    (int(xmin), int(ymin)),
                    (int(xmax), int(ymax)),
                    color=(255, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA

                )

                # cv2.circle(orig_image, (int(np.mean(face_points_x)), int(np.mean(face_points_y))), radius=1, color=(0, 0, 255), thickness=3)
            # Convert from BGR to RGB color format.
            # cv2.imshow('image', orig_image)
            out.write(orig_image)
            # Press `q` to exit.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release VideoCapture().
    cap.release()
    # Close all frames and video windows.
    cv2.destroyAllWindows()
