# -*- coding: utf-8 -*-
import pyrealsense2 as rs
import numpy as np
import cv2
 
 
''' 
设置
'''
pipeline = rs.pipeline()    # 定义流程pipeline，创建一个管道
config = rs.config()    # 定义配置config
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)      # 配置depth流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)     # 配置color流

# config.enable_stream(rs.stream.depth,  848, 480, rs.format.z16, 90)
# config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

# config.enable_stream(rs.stream.depth,  1280, 720, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

pipe_profile = pipeline.start(config)       # streaming流开始

pc = rs.pointcloud()        # 声明点云对象
points = rs.points()
 

''' 
获取图像帧
'''
def get_images():
    
    frames = pipeline.wait_for_frames()     # 等待获取图像帧，获取颜色和深度的框架集     

    depth_frame = frames.get_depth_frame()      # 获取depth帧 
    color_frame = frames.get_color_frame()      # 获取color帧

    ###### 将images转为numpy arrays #####  
    img_color = np.asanyarray(color_frame.get_data())       # RGB图  
    img_depth = np.asanyarray(depth_frame.get_data())       # 深度图（默认16位）

    return  img_color, img_depth, depth_frame, color_frame


''' 
获取随机点三维坐标(点云方法)
'''
def get_3d_camera_coordinate(depth_pixel, color_frame, depth_frame):
    x = depth_pixel[0]
    y = depth_pixel[1]

    ###### 计算点云 #####
    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)
    vtx = np.asanyarray(points.get_vertices())
    #  print ('vtx_before_reshape: ', vtx.shape)        # 307200
    vtx = np.reshape(vtx,(480, 640, -1))   
    # print ('vtx_after_reshape: ', vtx.shape)       # (480, 640, 1)

    camera_coordinate = vtx[y][x][0]
    # print ('camera_coordinate: ',camera_coordinate)
    dis = camera_coordinate[2]
    return dis, camera_coordinate



if __name__=="__main__":
    while True:
        ''' 
        获取图像帧
        '''
        img_color, img_depth, depth_frame, color_frame = get_images()        # 获取图像


        ''' 
        获取随机点三维坐标
        '''
        depth_pixel = [320, 240]        # 设置随机点，以相机中心点为例
        dis, camera_coordinate = get_3d_camera_coordinate(depth_pixel, color_frame, depth_frame)
        # print ('depth: ',dis)       # 深度单位是m
        # print ('camera_coordinate: ',camera_coordinate)


        ''' 
        显示图像与标注
        '''
        #### 在图中标记随机点及其坐标 ####
        cv2.circle(img_color, (320,240), 8, [255,0,255], thickness=-1)
        cv2.putText(img_color,"Dis:"+str(dis)+" m", (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[0,0,255])
        cv2.putText(img_color,"X:"+str(camera_coordinate[0])+" m", (80,80), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[255,0,0])
        cv2.putText(img_color,"Y:"+str(camera_coordinate[1])+" m", (80,120), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[255,0,0])
        cv2.putText(img_color,"Z:"+str(camera_coordinate[2])+" m", (80,160), cv2.FONT_HERSHEY_SIMPLEX, 1.2,[255,0,0])
        
        #### 显示画面 ####
        cv2.imshow('RealSence',img_color)
        key = cv2.waitKey(1)
