import os
import rosbag
from rs_utils import get_topic_table
import numpy as np
import cv2
import mediapipe as mp
import pandas as pd
import pyrealsense2 as rs

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

EDGES = list(mp_pose.POSE_CONNECTIONS)
WIDTH = 640
HEIGHT = 480
ROTATION = [0.999975, -0.00439544, -0.00559921, 0.00437222, 0.999982, -0.00415136, 0.00561735, 0.00412678, 0.999976]
TRANSLATION = [0.0147789, 4.98384e-05, 0.000502731]

def depth_rgb_registration(depthData, rgbData,intrinsic_d, intrinsic_rgb):
    depthScale = 1
    fx_d, fy_d, cx_d, cy_d = intrinsic_d[0,0], intrinsic_d[1,1], intrinsic_d[0,2], intrinsic_d[1,2]
    fx_rgb, fy_rgb, cx_rgb, cy_rgb = intrinsic_rgb[0,0], intrinsic_rgb[1,1], intrinsic_rgb[0,2], intrinsic_rgb[1,2]
    depthHeight = depthData.shape[0]
    depthWidth = depthData.shape[1]
    rgbHeight = rgbData.shape[0]
    rgbWidth = rgbData.shape[1]
    rotation = np.asanyarray(ROTATION).reshape((3,3))
    translation = np.asanyarray(TRANSLATION).reshape((3,1))
    extrinsics = np.zeros((4,4))
    extrinsics[:3,:3] = rotation
    extrinsics[:3,3] = translation.flatten()
    extrinsics[3,3] = 1
    aligned = np.zeros((depthHeight, depthWidth, 6), dtype=np.float32)

    for v in range(depthHeight):
        for u in range(depthWidth):
            z = depthData[v,u] / depthScale
            x = (u - cx_d) * z / fx_d
            y = (v - cy_d) * z / fy_d
            
            transformed = np.dot(extrinsics, np.array([x, y, z, 1]))
            aligned[v,u,:3] = transformed[:3]

    for v in range(depthHeight):
        for u in range(depthWidth):
            x = aligned[v,u,0] * fx_rgb / aligned[v,u,2] + cx_rgb
            y = aligned[v,u,1] * fy_rgb / aligned[v,u,2] + cy_rgb
            
            if (x >= rgbWidth or y >= rgbHeight or x < 0 or y < 0 or np.isnan(x) or np.isnan(y)):
                continue
            
            x = int(round(x))
            y = int(round(y))
            
            aligned[v,u,3:] = rgbData[y, x]
    # print(aligned[:,:,0] == aligned[:,:,1], aligned[:,:,0] == aligned[:,:,2])
    return aligned[:,:,:3]


def extract_frames():
    path = os.path.join('Measures', 'test', 'Movement', 'Baseline', 'RealSense')
    reader = rosbag.Bag(os.path.join(path, 'test.bag'))
    table = get_topic_table(reader)
    table.to_csv(os.path.join(path, 'topic.csv') )
    depth_info = table[(table['Topics'].str.contains('Depth')) & table['Topics'].str.contains('camera_info')]
    depth_info = list(depth_info['Topics'])
    topic_table = table[(table['Types'].str.contains('sensor_msgs')) & (table['Message Count'] > 1)]
    topics = list(topic_table['Topics'])
    for topic, msg, t in reader.read_messages(topics=topics):
        ts = msg.header.stamp.to_nsec()
        frame_id = msg.header.seq
        frame_id = f'{frame_id:0{3}d}'
        img_data = msg.data
        width, height = msg.width, msg.height
        if 'Color' in topic:
            img = np.frombuffer(img_data, dtype = np.uint8).reshape((height, width, 3))
            cv2.imwrite(os.path.join(path, 'Color', f'FrameID_{frame_id}.png'), img)
        if 'Depth' in topic:
            img = np.frombuffer(img_data, dtype = np.uint16).reshape((height, width))
            cv2.imwrite(os.path.join(path, 'Depth', f'FrameID_{frame_id}.png'), img)

def show_color_depth():
    video = 'test'
    path = os.path.join('Measures', video, 'Movement', 'Baseline', 'RealSense')
    color_list = sorted(os.listdir(os.path.join(path, 'Color')))
    depth_list = sorted(os.listdir(os.path.join(path, 'Depth')))
    skeleton_list =os.listdir(os.path.join(path, 'Skeletons'))
    skeleton_list = sorted(skeleton_list, key=lambda x: int(x.split('_')[1]))
    color_id = [int(img.split('_')[1].split('.')[0]) for img in color_list]
    depth_id = [int(img.split('_')[1].split('.')[0]) for img in depth_list]
    skeleton_id = [int(img.split('_')[1]) for img in skeleton_list]
    reader = rosbag.Bag(os.path.join(path, f'{video}.bag'))
    topic_table = get_topic_table(reader)
    depth_info = topic_table[(topic_table['Topics'].str.contains('Depth')) & (topic_table['Topics'].str.contains('camera_info'))]
    depth_info = list(depth_info['Topics'])
    for top, msg, t in reader.read_messages(topics=depth_info):
        intrinsic_matrix_color = np.asanyarray(msg.K).reshape((3,3))
    depth_info = topic_table[(topic_table['Topics'].str.contains('Color')) & (topic_table['Topics'].str.contains('camera_info'))]
    depth_info = list(depth_info['Topics'])
    for top, msg, t in reader.read_messages(topics=depth_info):
        intrinsic_matrix_depth = np.asanyarray(msg.K).reshape((3,3))
    translation_depth_to_color = np.array([0.014778909273445606, 4.9838410632219166e-05, 0.0005027310107834637])
    # print(intrinsic_matrix_color, intrinsic_matrix_color.shape)
    for idx, img in enumerate(color_list):
        id = color_id[idx]
        if id in depth_id and id in skeleton_id:
            skeleton = os.path.join(path, 'Skeletons', skeleton_list[skeleton_id.index(id)])
            color = os.path.join(path, 'Color', color_list[idx])
            depth = os.path.join(path,'Depth', depth_list[depth_id.index(id)])
            color = cv2.imread(color)
            depth = cv2.imread(depth, cv2.IMREAD_GRAYSCALE)
            # depth = align_depth(intrinsic_matrix_color, intrinsic_matrix_depth, depth, translation_depth_to_color)
            depth = depth_rgb_registration(depth, color, intrinsic_matrix_depth, intrinsic_matrix_color)
            # depth = cv2.resize(depth, (color.shape[1], color.shape[0]))
            depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
            draw_skeleton(color, skeleton)
            draw_skeleton(depth,skeleton)
            img = np.hstack([color, depth])
            cv2.imshow('AAA', img)
            cv2.waitKey(1)
    cv2.destroyAllWindows()

def draw_skeleton(img, skel):
    joints = pd.read_csv(skel)
    for _, row in joints.iterrows():
        if row['confidence'] > 0.5:
            x = int(row['x-norm']*WIDTH)
            y = int(row['y-norm']*HEIGHT)
            cv2.circle(img, (x,y), radius=5, color = (0,0,255), thickness=-1)
        for edge in EDGES:
            row1 = joints.iloc[edge[0]]
            row2 = joints.iloc[edge[1]]
            x1, y1 = int(row1['x-norm']*WIDTH), int(row1['y-norm']*HEIGHT)
            x2, y2 = int(row2['x-norm']*WIDTH), int(row2['y-norm']*HEIGHT)
            cv2.line(img, (x1, y1), (x2, y2), color = (255, 0, 0))

def mediapipe_depth():
    path = os.path.join('Measures', 'test','Movement', 'Baseline', 'RealSense')
    depth_list = sorted(os.listdir(os.path.join(path, 'Depth')))
    with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
        for depth in depth_list:
            img = cv2.imread(os.path.join(path, 'Depth', depth))
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_8U)
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            img.flags.writeable = False
            results = pose.process(img)
            img.flags.writeable = True
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow('aaa', img)
            cv2.waitKey(1)
        cv2.destroyAllWindows()
# extract_frames()
show_color_depth()