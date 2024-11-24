import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import cvzone
import math
import time
import logging

logging.getLogger("ultralytics").setLevel(logging.ERROR)  # Hide non-error logs

# RealSense setup
pipeline = rs.pipeline()  # Define the pipeline
config = rs.config()  # Define the configuration
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)  # Configure depth stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)  # Configure color stream
pipe_profile = pipeline.start(config)  # Start the stream

pc = rs.pointcloud()  # Declare point cloud object
points = rs.points()

# YOLO setup
model = YOLO(r"D:\postgruduate\face_robot\pickup\yolov8l.pt", verbose=False)
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


def get_images():
    frames = pipeline.wait_for_frames()  # Wait for the next set of frames
    depth_frame = frames.get_depth_frame()  # Get the depth frame
    color_frame = frames.get_color_frame()  # Get the color frame

    # Convert images to numpy arrays
    img_color = np.asanyarray(color_frame.get_data())
    img_depth = np.asanyarray(depth_frame.get_data())

    return img_color, img_depth, depth_frame, color_frame


def get_3d_camera_coordinate(depth_pixel, color_frame, depth_frame):
    x = depth_pixel[0]
    y = depth_pixel[1]

    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)
    vtx = np.asanyarray(points.get_vertices())
    vtx = np.reshape(vtx, (480, 640, -1))

    camera_coordinate = vtx[y][x]
    return camera_coordinate


if __name__ == "__main__":
    while True:
        # Capture RealSense frames
        img_color, img_depth, depth_frame, color_frame = get_images()

        # Detect objects using YOLO
        results = model(img_color, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding box details
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Calculate the center of the bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Get 3D coordinates of the center
                depth_pixel = [center_x, center_y]
                camera_coordinate = get_3d_camera_coordinate(depth_pixel, color_frame, depth_frame)

                # Confidence and class name
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]

                # Display bounding box, confidence, and class
                cvzone.cornerRect(img_color, (x1, y1, w, h))
                cvzone.putTextRect(img_color, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

                # Display 3D coordinates
                cv2.putText(img_color, f"X: {camera_coordinate[0]:.2f} m", (x1, y1 - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(img_color, f"Y: {camera_coordinate[1]:.2f} m", (x1, y1 - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(img_color, f"Z: {camera_coordinate[2]:.2f} m", (x1, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display image
        cv2.imshow("RealSense + YOLO", img_color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    pipeline.stop()
    cv2.destroyAllWindows()
