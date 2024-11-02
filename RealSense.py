import pyrealsense2 as rs
import mediapipe as mp
import os
import numpy as np
import cv2
import pandas as pd
import rosbag
import threading
from datetime import datetime, timezone

from utils.rs_utils import *

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils



# print(BODY_PARTS_MAPPING[0], BODY_PARTS_MAPPING['NOSE'])
class RealSense():
    def __init__(self,
                 tag_text = 'default_RealSense',
                 save_rs_data = False,
                 connect_rs = False,
                 frame_rate = 30,
                 enable_color = True,
                 enable_depth = True,
                 enable_ir = False):
        
        self.tag_text = tag_text
        self.ts = []

        self.stop_rs_event = threading.Event()
        self.rs_lock = threading.Lock()

        self.frame_rate = frame_rate
        self.save_rs_data = save_rs_data
        self.connect_rs = connect_rs
        self.is_connected=False
        # self.save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Measures", self.tag_text, 'RealSense')
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.enable_ir = enable_ir

        self.stop_event = threading.Event()
        self.pose = mp_pose.Pose(min_detection_confidence = .5, min_tracking_confidence =.5) 
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        # self.pipe = rs.pipeline()
        # self.cfg = None
        # self.cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.frame_rate)
        # self.cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.frame_rate)
        # self.cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, self.frame_rate)
        # self.pc = rs.pointcloud()
    
    def create_paths(self):
        self.color_path = os.path.join(self.save_path, 'Color')
        self.depth_path = os.path.join(self.save_path, 'Depth')
        self.points_path = os.path.join(self.save_path, 'PointCloud')
        self.skeleton_path = os.path.join(self.save_path, 'Skeletons')
        self.ir_path = os.path.join(self.save_path, 'InfraRed')
        # self.dir_log = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Measures', self.tag_text, 'Log_Files')
        self.dir_log = os.path.join('Measures', self.tag_text,'Log_Files' )
        self.bagName = os.path.join(self.save_path,f'{self.tag_text}.bag')
        if not os.path.exists(self.dir_log): os.makedirs(self.dir_log)
        if not os.path.exists(self.save_path) : os.makedirs(self.save_path)
        if self.enable_color:
            if not os.path.exists(self.color_path) : os.makedirs(self.color_path)
        if self.enable_depth:
            if not os.path.exists(self.depth_path) : os.makedirs(self.depth_path)
            if not os.path.exists(self.points_path) : os.makedirs(self.points_path)
        if not os.path.exists(self.skeleton_path) : os.makedirs(self.skeleton_path)
        if self.enable_ir:
            if not os.path.exists(self.ir_path) : os.makedirs(self.ir_path)
    def ConnectRealSense(self):
        self.pipe = rs.pipeline()

        self.is_connected = True
        with self.rs_lock:
            self.cfg = rs.config()
            if self.save_rs_data:
                self.cfg.enable_record_to_file(self.bagName)
            if self.enable_color:
                self.cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.frame_rate)
            if self.enable_depth:
                self.cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.frame_rate)
            if self.enable_ir:
                self.cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, self.frame_rate)
                self.cfg.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, self.frame_rate)
            try:
                self.pipe.start(self.cfg)
                _ = self.pipe.wait_for_frames()
                # timestamp =  int(datetime.now(timezone.utc).timestamp() * 1000)
                # print( int(datetime.now(timezone.utc).timestamp() * 1000), frames.get_color_frame().timestamp,)
                # with open(os.path.join(self.dir_log, f'{self.tag_text}_Events.txt'),'a') as file:
                # with open(self.connection_log, 'a') as file:
                #     # print('im here')
                #     file.write(f'Realsense-start, {timestamp} \n')
            except RuntimeError:
                tag_text = self.tag_text
                self.pipe.stop()
                print('Didnt get Infrared Frames. Disabled Infrared stream')
                # self.status_realsense.config(text = 'Connected without IR')
                message = f'Connecting Realsense with Measurement Tag: {tag_text}...'
                self.cfg = rs.config()
                print(message)
                if self.enable_color:
                    self.cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.frame_rate)
                if self.enable_depth:
                    self.cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.frame_rate)
                self.pipe.start(self.cfg)
                _ = self.pipe.wait_for_frames()
                timestamp =  int(datetime.now(timezone.utc).timestamp() * 1000)
                # print( int(datetime.now(timezone.utc).timestamp() * 1000), frames.get_color_frame().timestamp,)
                # with open(os.path.join(self.dir_log, f'{self.tag_text}_Events.txt'),'a') as file:
                with open(self.connection_log, 'a') as file:
                    file.write(f'Realsense-start without IR, {timestamp} \n')
            except Exception as e:
                print(f'Error connecting RealSense : {e}')
        
                    # self.update_terminal(message)
    def get_data(self, show_capture = True):
        try:
            with self.rs_lock:
                if self.is_connected:
                    frames = self.pipe.wait_for_frames()
                    frames = self.align.process(frames)
                    color_frame = frames.get_color_frame()
                    timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
                    # print(timestamp, color_frame.timestamp)
                    self.ts.append([timestamp, color_frame.timestamp])
                    color_data = np.asanyarray(color_frame.get_data())
                    color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
                    color_data.flags.writeable = False
                    results = self.pose.process(color_data)
                    color_data.flags.writeable = True
                    color_data = cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR)
                    # timestamp = color_frame.timestamp
                    skel_file = os.path.join(self.skeleton_path,f'FrameID_{color_frame.frame_number}_TimeStamp_{timestamp}.csv')
                    if results.pose_landmarks:
                        write_skeleton_data(skel_file, results.pose_landmarks)
                        if show_capture:
                            mp_drawing.draw_landmarks(color_data, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    # print(f'Acquired skeleton frame number {color_frame.frame_number} at timestamp: {timestamp}')
                    # if show_capture:
                    #     cv2.imshow('RGB Stream', color_data)
                    #     cv2.waitKey(1)
                    return color_data, self.ts

                else:
                    self.pipe.stop()
                    # cv2.destroyAllWindows()
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     return
        except Exception as e:
            self.pipe.stop()
            print(f'Error Realsense : {e}')

    def start_streaming(self, show_capture = True):
        self.pipe = rs.pipeline()

        self.is_connected = True
        self.cfg = rs.config()
        self.cfg.enable_record_to_file(self.bagName)

        if self.enable_color:
            self.cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.frame_rate)
        if self.enable_depth:
            self.cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.frame_rate)
        if self.enable_ir:
            self.cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, self.frame_rate)
            self.cfg.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, self.frame_rate)

        self.pipe.start(self.cfg)
        align_to = rs.stream.color
        align = rs.align(align_to)
        
        with mp_pose.Pose(min_detection_confidence = .5, min_tracking_confidence =.5) as pose:
            try:
                while True:
                    if self.is_connected == False:
                        self.interrupted = True
                    frames = self.pipe.wait_for_frames()
                    frames = align.process(frames)
                    color_frame = frames.get_color_frame()
                    color_data = np.asanyarray(color_frame.get_data())
                    color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
                    color_data.flags.writeable = False
                    results = pose.process(color_data)
                    color_data.flags.writeable = True
                    color_data = cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR)
                    skel_file = os.path.join(self.skeleton_path,f'FrameID_{color_frame.frame_number}_TimeStamp_{color_frame.timestamp}.csv')

                    if results.pose_landmarks:
                        if self.save_rs_data:
                           write_skeleton_data(skel_file, results.pose_landmarks)
                        if show_capture:
                            mp_drawing.draw_landmarks(color_data, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    if show_capture:
                        cv2.imshow('RGB Stream', color_data)
                    # print(f'Acquired skeleton frame number {color_frame.frame_number} at timestamp: {color_frame.timestamp}')
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            # except KeyboardInterrupt:
            #     pass
            except Exception as e:
                print(e)
                self.interrupted = True
            finally:
                cv2.destroyAllWindows()
                self.pipe.stop()
                print('Realsense disconnected')
                # self.stop_streaming()
                self.bag_writer.close()
    def stop_streaming(self):
        self.stop_event.set()

    def reset_streaming(self):
        self.stop_event.clear()
        threading.Thread(target = self.start_streaming).start()

    def post_processing_extraction(self):
        reader = rosbag.Bag(self.bagName)
        topic_table = get_topic_table(reader)
        topic_table.to_csv(os.path.join(self.save_path, f'{self.tag_text}_topics.csv'))
        depth_info = topic_table[(topic_table['Topics'].str.contains('Depth')) & (topic_table['Topics'].str.contains('camera_info'))]
        depth_info = list(depth_info['Topics'])
        for top, msg, t in reader.read_messages(topics=depth_info):
            intrinsic_matrix = msg.K
        intrinsic_matrix = np.asanyarray(intrinsic_matrix).reshape((3,3))
        topic_table = topic_table[(topic_table['Types'].str.contains('sensor_msgs')) & (topic_table['Message Count'] > 1)]
        topics = list(topic_table['Topics'])

        for topic, msg, t in reader.read_messages(topics = topics):
            timestamp = (msg.header.stamp.to_nsec())
            frame_id = msg.header.seq
            img_data = msg.data
            width = msg.width
            height = msg.height
        
            if 'Color' in topic:
                img_array = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width, 3))
                cv2.imwrite(os.path.join(self.color_path,f'FrameID_{frame_id}_TimeStamp_{timestamp}.png'), img_array)
                print(f'Saved color information at frame {frame_id}')
            
            if 'Depth' in topic:
                img_array = np.frombuffer(img_data, dtype = np.uint16).reshape((height,width))
                cv2.imwrite(os.path.join(self.depth_path,f'FrameID_{frame_id}_TimeStamp_{timestamp}.png'), img_array)
                points = depth_to_pc(img_array, intrinsic_matrix)
                # points = self.pc(get_depth_frame(img_array))
                # points = self.pc.calculate(img_data)
                # vert, text = points.get_vertices(), points.get_texture_coordinates()
                # points_data = np.asanyarray(vert).view(np.float32).reshape(-1,3)
                # texcoord = np.asanyarray(text).view(np.float32).reshape(-1,2)
                np.save(os.path.join(self.points_path,f'frameID_{frame_id}_TimeStamp_{timestamp}.npy'), points)
                print(f'Saved depth information at frame {frame_id}')

            if 'Infrared_1' in topic:
                img_array = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width))
                cv2.imwrite(os.path.join(self.ir_path,f'FrameID_{frame_id}_TimeStamp_{timestamp}_IR1.png'), img_array)
                print(f'Saved ir1 information at frame {frame_id}')

            if 'Infrared_2' in topic:
                img_array = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width))
                cv2.imwrite(os.path.join(self.ir_path,f'FrameID_{frame_id}_TimeStamp_{timestamp}_IR2.png'), img_array)
                print(f'Saved ir2 information at frame {frame_id}')

def main():
    print('aa')
    try:
        realSense = RealSense(save_rs_data=True,
                            connect_rs= True,
                            tag_text='testAligned')
        realSense.save_path = 'Measures'
        realSense.create_paths()
        # realSense.start_streaming()
    except Exception as e:
        print(e)
    realSense.post_processing_extraction()

if __name__ == '__main__':
    main()