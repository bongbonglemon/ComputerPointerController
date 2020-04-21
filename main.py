import os
import sys
import time
import socket
import json
import cv2
import numpy as np
import time

from face_detection import FaceDetect
from input_feeder import InputFeeder
from facial_landmarks_detection import EyesDetect
from head_pose_estimation import AngleDetect
from gaze_estimation import GazeDetect
from mouse_controller import MouseController

from argparse import ArgumentParser


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    
    parser = ArgumentParser()
    parser.add_argument("-f", "--face", required=False, type=str,
                        help="Path to an xml file with a face detection model.")
    parser.add_argument("-e", "--eyes", required=False, type=str,
                        help="Path to an xml file with a landmark detection model.")
    parser.add_argument("-a", "--angle", required=False, type=str,
                        help="Path to an xml file with a head pose estimation model.")
    parser.add_argument("-g", "--gaze", required=False, type=str,
                        help="Path to an xml file with a gaze estimation model.")
    parser.add_argument("-v", "--video", required=False, default='video', type=str,
                        help="Toggle prepared video or webcam stream.")
    parser.add_argument("-i", "--input", required=False, type=str,
                        help="Path to image or video file")

    return parser

def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()

    start_time = time.time()
    face_detector = FaceDetect(model_name=args.face)
    face_detector.load_model()
    print("Time taken to load face detection model (in seconds):", time.time()-start_time)

    start_time = time.time()
    eyes_detector = EyesDetect(model_name=args.eyes)
    eyes_detector.load_model()
    print("Time taken to load landmark detection model (in seconds):", time.time()-start_time)

    start_time = time.time()
    angle_detector = AngleDetect(model_name=args.angle)
    angle_detector.load_model()
    print("Time taken to load head pose estimation model (in seconds):", time.time()-start_time)

    start_time = time.time()
    gaze_detector = GazeDetect(model_name=args.gaze)
    gaze_detector.load_model()
    print("Time taken to load gaze estimation model (in seconds):", time.time()-start_time)

    mouse_controller = MouseController('medium','medium')
    
    feed=InputFeeder(input_type=args.video, input_file=args.input)
    feed.load_data()
    for batch in feed.next_batch():
        face = face_detector.predict(batch)
        left_eye, right_eye = eyes_detector.predict(face)
        angles = angle_detector.predict(face)
        x, y = gaze_detector.predict(left_eye, right_eye, angles)
        mouse_controller.move(x, y)
        
    feed.close()
    


if __name__ == '__main__':
    main()
