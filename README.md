# basketball detection and counting project
This repository is for 2110433 Computer Vision course Final Project


### Problem Statement
In basketball training ,manually tracking a single basketball's movement during drills or practice sessions is time-consuming. We aim to develop a computer vision system to automatically detect and count the basketball's movements in real-time video streams, improving efficiency.
### Technical Challenges
- model accuracy
  
  Based on our solution, we need to determine the position of the basketball relative to the rim, both above and below, to predict whether the basketball successfully enters the goal or not.
- time/resource for training

  Each training session spans approximately 80 epochs and consumes around 400 minutes of computation time on a MacBook Pro M2, which has hardware limitations.
### Related Works
- https://github.com/chonyy/AI-basketball-analysis

  The object detection model is trained with the Faster R-CNN model architecture, which includes pretrained weight on COCO dataset.
- https://github.com/TwinKay/AI_BasketBall_Video_Analysis

  The object detection model is trained with the YOLO-NAS-L, Person Re-Identification is trained with  MobileNetV3 for faster inference time.
### Method and Results
- YOLOv8 without Region of interest (ROI)
- YOLOv8 with Region of interest (ROI)
### Discussion and Future Work
- Due to hardware limitations, it is hard to process real-time data
- Need more sample images to train our model to increase its accuracy for various basketballs and hoops
- The current model may detect other object as a basketball which can be fixed by finetuning with negative sample images
### Future Work
- Person Re-Identification: Calculate the accuracy of each player's shooting during a game

- Pose estimation analysis: Calculate the likelihood for each player based on their shooting form

- Optimize the model: Use ONNX to convert the model weights for faster inference times, enabling real-time video processing
