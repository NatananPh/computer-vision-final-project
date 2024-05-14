import math
import cv2
import cvzone
import argparse
import numpy as np
import streamlit as st
from os.path import exists

from ultralytics import YOLO
from utils import (
    score,
    detect_down,
    detect_up,
    in_hoop_region,
    clean_hoop_pos,
    clean_ball_pos,
    resize_with_pad,
)


def validate_file_path(file_path):
    return exists(file_path)


def video_input(data_source):
    if data_source == 'Sample data':
        video_path = "/sample/two_score_two_miss.mp4"
        return video_path
    else:
        uploaded_file = st.sidebar.file_uploader("Choose a video", type=['mp4', 'mpv', 'avi'])

        # To read file as bytes:
        if uploaded_file is not None:
            st.write("filename:", uploaded_file.name)
            with open(f'./sample/{uploaded_file.name}', 'wb') as file:
                file.write(uploaded_file.read())

        # path to saved file
        return f'./sample/{uploaded_file.name}'


class ShotDetector:
    def __init__(self, video_path, confidence=0.15, save_video=False):
        self.video_path = video_path
        self.save_video = save_video
        self.confidence = confidence

        # video_path = 0 to use webcam (streamed on iPhone using Iriun Webcam)
        if self.video_path != 0:
            if '/' in self.video_path:
                self.output_vdo_name = f"{self.video_path.split('/')[-1].split('.')[0]}"
            else:
                self.output_vdo_name = self.video_path.split('.')[0]
        else:
            self.output_vdo_name = f"webcam"

        if self.save_video:
            self.output_writer = cv2.VideoWriter(
                f"./sample/{self.output_vdo_name}_result.mp4",
                cv2.VideoWriter_fourcc(*'mp4v'), 
                60, (640, 640)
            )

        # Load the YOLO model created from main.py - change text to your relative path
        self.model = YOLO("best.pt")
        self.class_names = ["Basketball", "Basketball Hoop"]

        # Use video - replace text with your video path
        self.cap = cv2.VideoCapture(self.video_path)

        # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
        self.ball_pos = ([])
        self.hoop_pos = ([])

        self.frame_count = 0
        self.frame = 60

        self.makes = 0
        self.attempts = 0

        # Used to detect shots (upper and lower region)
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0

        # Used for green and red colors after make/miss
        self.fade_frames = 30
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

        # run if file is in dir
        if validate_file_path(self.video_path):
            print(f"Found {self.video_path}")
            self.run()
        else:
            st.write(f"{self.video_path} NOT found")


    def run(self):
        st1, st2, st3 = st.columns(3)
        with st1:
            st.markdown("## Frame count:")
            st1_text = st.markdown(f"{self.frame_count}")
        with st2:
            st.markdown("## Score:")
            st2_text = st.markdown(f"{0}")
        with st3:
            st.markdown("## Attempts:")
            st3_text = st.markdown(f"{0}")

        st.markdown("---")
        output = st.empty()

        while True:
            success, self.frame = self.cap.read()

            # End of the video or an error occurred
            if not success:
                st.write("Can't read frame....")
                break

            self.frame = resize_with_pad(self.frame, (640, 640))
            results = self.model(self.frame, stream=True)

            for r in results:
                boxes = r.boxes

                # Bounding box
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # Class Name
                    cls = int(box.cls[0])
                    current_class = self.class_names[cls]

                    center = (int(x1 + w / 2), int(y1 + h / 2))

                    # Only create ball points if high confidence or near hoop
                    if (in_hoop_region(center, self.hoop_pos) and conf > self.confidence) and \
                        current_class == "Basketball":
                        self.ball_pos.append((center, self.frame_count, w, h, conf))
                        cvzone.cornerRect(self.frame, (x1, y1, w, h))

                    # Create hoop points if high confidence
                    if conf > 0.3 and current_class == "Basketball Hoop":
                        self.hoop_pos.append((center, self.frame_count, w, h, conf))
                        cvzone.cornerRect(self.frame, (x1, y1, w, h))

            self.clean_motion()
            self.shot_detection()
            self.fade_color_after_shot()
            self.frame_count += 1

            # writing the video frame (.mp4)
            if self.save_video:
                self.output_writer.write(self.frame)
            
            # Stream results
            output.image(self.frame[:, :, ::-1])
            st1_text.markdown(f"### **{self.frame_count}**")
            st2_text.markdown(f"### **{self.makes}**")
            st3_text.markdown(f"### **{self.attempts}**")
        
        if self.save_video:
            self.output_writer.release()
        self.cap.release()
        print(f"Finish Detection in ./sample/{self.output_vdo_name}_result.mp4" if self.save_video else "Finish Detection..")


    # Clean and display ball motion
    def clean_motion(self):
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        for i in range(0, len(self.ball_pos)):
            cv2.circle(self.frame, self.ball_pos[i][0], 2, (0, 0, 255), 2)

        # Clean hoop motion and display current hoop center
        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (128, 128, 0), 2)


    # Detecting when ball is in 'up' and 'down' area - ball can only be in 'down' area after it is in 'up'
    def shot_detection(self):
        if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
            if not self.up:
                self.up = detect_up(self.ball_pos, self.hoop_pos)
                if self.up:
                    self.up_frame = self.ball_pos[-1][1]

            if self.up and not self.down:
                self.down = detect_down(self.ball_pos, self.hoop_pos)
                if self.down:
                    self.down_frame = self.ball_pos[-1][1]

            # If ball goes from 'up' area to 'down' area in that order, increase attempt and reset
            if self.frame_count % 10 == 0:
                if self.up and self.down and self.up_frame < self.down_frame:
                    self.attempts += 1
                    self.up = False
                    self.down = False

                    # If it is a make, put a green overlay
                    if score(self.ball_pos, self.hoop_pos):
                        self.makes += 1
                        self.overlay_color = (0, 255, 0)
                        self.fade_counter = self.fade_frames

                    # If it is a miss, put a red overlay
                    else:
                        self.overlay_color = (0, 0, 255)
                        self.fade_counter = self.fade_frames


    # Gradually fade out color after shot
    def fade_color_after_shot(self):
        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            self.frame = cv2.addWeighted(
                self.frame, 1 - alpha,
                np.full_like(self.frame, self.overlay_color),
                alpha, 0
            )
            self.fade_counter -= 1


if __name__ == "__main__":
    st.title("Basketball Detection Dashboard")
    st.sidebar.title("Settings")

    # confidence slider
    confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=.15)

    save_video = st.sidebar.checkbox("Save video output", value=False)
    data_source = st.sidebar.radio("Select input source: ", ['Sample data', 'Upload your own data'])
    upload_file = video_input(data_source)
    ShotDetector(upload_file, confidence=confidence, save_video=save_video)