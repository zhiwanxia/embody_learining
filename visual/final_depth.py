from ultralytics import YOLO
import cv2
import math
import numpy as np
import pyrealsense2 as rs
import logging

# 隐藏非错误日志
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# 加载 YOLO 模型
model = YOLO(r"D:\postgruduate\face_robot\embody_learning\visual\yolov8l.pt", verbose=False)

# 定义 COCO 数据集的类别名称
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# RealSense 设置
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe_profile = pipeline.start(config)

# 创建对齐对象
align_to = rs.stream.color
align = rs.align(align_to)


def get_aligned_images():
    """获取对齐后的彩色图和深度图"""
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    aligned_depth_frame = aligned_frames.get_depth_frame()
    aligned_color_frame = aligned_frames.get_color_frame()

    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    img_color = np.asanyarray(aligned_color_frame.get_data())
    img_depth = np.asanyarray(aligned_depth_frame.get_data())

    return depth_intrin, img_color, img_depth, aligned_depth_frame


def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin):
    """通过像素点和深度图计算 3D 坐标"""
    x, y = depth_pixel
    dis = aligned_depth_frame.get_distance(x, y)  # 获取深度值（单位：米）
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], dis)
    return dis, camera_coordinate


if __name__ == "__main__":
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  
    while True:
        # 获取对齐后的图像
        depth_intrin, img_color, img_depth, aligned_depth_frame = get_aligned_images()

        # YOLO 检测物体
        results = model(img_color, stream=True, conf=0.6)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # 获取检测框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 检测框坐标
                w, h = x2 - x1, y2 - y1
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2  # 检测框中心点

                # 获取物体类别和置信度
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                # 计算中心点的三维坐标
                dis, camera_coordinate = get_3d_camera_coordinate([center_x, center_y], aligned_depth_frame, depth_intrin)

                # 绘制普通矩形框和三维坐标
                cv2.rectangle(img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 使用绿色矩形框
                label = f'{classNames[cls]} {conf}'
                xyz_label = f" {camera_coordinate[0]:.2f},{camera_coordinate[1]:.2f}, {camera_coordinate[2]:.2f}"

                # 在检测框顶部显示类别和三维坐标信息
                cv2.putText(img_color, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(img_color, xyz_label, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 显示检测结果
        cv2.imshow("Image", img_color)
        key = cv2.waitKey(1)

        # 按 'q' 键退出程序
        if key & 0xFF == ord('q'):
            break

    # 停止 RealSense 管道
    pipeline.stop()
    cv2.destroyAllWindows()
