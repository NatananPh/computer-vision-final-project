import math
import cv2
import cvzone
import argparse
import numpy as np

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


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', type=str, default=None, help='video/0 for webcam')
    parser.add_argument('-vi', '--view-img', action='store_true', help='display results')
    parser.add_argument('-sv', '--save-video', action='store_true', help='save video result')
    opt = parser.parse_args()
    return opt


class ShotDetector:
    def __init__(self, video_path, view_image=False, save_video=False):
        self.video_path = video_path
        self.view_image = view_image
        self.save_video = save_video

        # video_path = 0 to use webcam (streamed on iPhone using Iriun Webcam)
        if self.video_path != 0:
            if '/' in self.video_path:
                self.output_vdo_name = f"{self.video_path.split('/')[-1].split('.')[0]}"
            else:
                self.output_vdo_name = self.video_path.split('.')[0]
        else:
            self.output_vdo_name = f"webcam_result"

        self.output_writer = cv2.VideoWriter(
            f"./sample/{self.output_vdo_name}.mp4",
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

        self.run()


    def run(self):
        while True:
            success, self.frame = self.cap.read()
            self.frame = resize_with_pad(self.frame, (640, 640))

            # End of the video or an error occurred
            if not success:
                break

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
                    if (in_hoop_region(center, self.hoop_pos) and conf > 0.15) and \
                        current_class == "Basketball":
                        self.ball_pos.append((center, self.frame_count, w, h, conf))
                        cvzone.cornerRect(self.frame, (x1, y1, w, h))

                    # Create hoop points if high confidence
                    if conf > 0.3 and current_class == "Basketball Hoop":
                        self.hoop_pos.append((center, self.frame_count, w, h, conf))
                        cvzone.cornerRect(self.frame, (x1, y1, w, h))

            self.clean_motion()
            self.shot_detection()
            self.display_score()
            self.frame_count += 1

            # writing the video frame (.mp4)
            if self.save_video:
                self.output_writer.write(self.frame)
            
            # Stream results
            if self.view_image:
                cv2.imshow("Frame", self.frame)

                # Close if 'q' is clicked
                # higher waitKey slows video down, use 1 for webcam
                if cv2.waitKey(1) & 0xFF == ord("q"):  
                    break
        
        self.output_writer.release()
        self.cap.release()
        cv2.destroyAllWindows()
        print("Finish Detection" + f"in ./sample/{self.output_vdo_name}.mp4" if self.save_video else "..")


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


    # Add text
    def display_score(self):
        text = str(self.makes) + " / " + str(self.attempts)
        cv2.putText(self.frame,text, (50, 125), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3, (255, 255, 255), 6)
        cv2.putText(self.frame,text, (50, 125), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, (0, 0, 0), 3)

        # Gradually fade out color after shot
        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            self.frame = cv2.addWeighted(
                self.frame, 1 - alpha,
                np.full_like(self.frame, self.overlay_color),
                alpha, 0
            )
            self.fade_counter -= 1


if __name__ == "__main__":
    opt = parse_opt()
    view_image = opt.view_img
    save_video = opt.save_video

    # Get video source path (0 for webcam)
    if opt.source == None or opt.source == '0':
        video_path = 0
    else:
        video_path = opt.source

    ShotDetector(video_path, view_image, save_video)