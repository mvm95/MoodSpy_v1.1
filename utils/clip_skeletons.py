
from utils.visualizedata_bioharness import extract_pepper_timestamps
import os
import numpy as  np
import mediapipe as mp
import pandas as pd
import cv2
from tqdm import tqdm


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

edges = list(mp_pose.POSE_CONNECTIONS)

# print(edges)
measure_dir= '/home/marco/Desktop/Codes/MoodSpy/Measures'

def is_between(pair, ts):
    return pair[0] <= ts <= pair[1]

def create_clip(sub):
    pepper_log_file = os.path.join(measure_dir,sub,'Log_Files',f'{sub}_pepper.txt')
    timestamps_stop = extract_pepper_timestamps(pepper_log_file)
    # print(timestamps_stop)
    skeleton_folder = os.path.join(measure_dir,sub,'RealSense', 'Skeletons')
    skeleton_list = os.listdir(skeleton_folder)
    skeleton_list = sorted(skeleton_list, key=lambda x: int(x.split('_')[1]))
    # print(skeleton_list)

    height = 640
    width = 480

    start = timestamps_stop[0] - 3000 
    end_time = timestamps_stop[-1] + 3000
    timestamps_stop = [(timestamps_stop[i], timestamps_stop[i + 1]) for i in range(0, len(timestamps_stop), 2)]
    # delta di 3 secondi
    # print(start, end_time, timestamps_stop)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    font_color = (255, 255, 255) 
    frames=[]
    for skel in skeleton_list:
        ts = skel.split('_')[-1]
        ts = int(ts.split('.')[0])
        if ts >= start and ts <= end_time:
            # print(any(is_between(pair, ts) for pair in timestamps_stop))
            if not any(is_between(pair, ts) for pair in timestamps_stop):
                color = (0,255,0)
                text = 'GO'
            else:
                text = 'STOP'
                color = (0,0,255)
            black_image = np.zeros((height,width,3), dtype=np.uint8)
            joints = pd.read_csv(os.path.join(skeleton_folder,skel))
            for _,row in joints.iterrows():
                if row['confidence'] > 0.75:
                    x = int(row['x-norm']* width)
                    y = int(row['y-norm']* height)
                    cv2.circle(black_image, (x, y), radius=5, color=color, thickness=-1)
            for edge in edges:
                p1 = edge[0]
                p2 = edge[1]
                row1 = joints.iloc[p1]
                row2 = joints.iloc[p2]
                x1 = int(row1['x-norm']*width)
                y1 = int(row1['y-norm']*height)
                x2 = int(row2['x-norm']*width)
                y2 = int(row2['y-norm']*height)
                cv2.line(black_image, (x1,y1), (x2,y2), color = color)
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_position = ((black_image.shape[1] - text_size[0]) // 2, 40)  # Adjust the Y coordinate as needed

            cv2.putText(black_image, text, text_position, font, font_scale, font_color, font_thickness)
            frames.append(black_image)
            # cv2.imshow('clip', black_image)
            # cv2.waitKey(1)
    cv2.destroyAllWindows()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'XVID' or 'MJPG'
    video = cv2.VideoWriter(os.path.join(measure_dir,sub,f'{sub}.mp4'), fourcc, 10, (width, height))  # Adjust the frame rate (fps) as needed

    # Write each frame to the video
    for frame in frames:
        video.write(frame)

    # Release the video writer and close all OpenCV windows
    video.release()
    cv2.destroyAllWindows()
    
def main():
    for sub in tqdm(os.listdir(measure_dir)):
        print(sub)
        # if os.path.exists(os.path.join(measure_dir,sub,f'{sub}.mp4')):
        #     continue
        try:
            create_clip(sub)
        except Exception as e:
            print(e)

if __name__ == '__main__':
    # main()
    for sub in tqdm(os.listdir(measure_dir)):
                print(sub)
        # if 'IDU003V001' in sub or 'IDU002' in sub:
            # if not os.path.exists(f'/home/marco/Desktop/Codes/MoodSpy/Measures/{sub}/{sub}.mp4'):
                try:
                    create_clip(sub)
                except Exception as e: 
                    print(e)
    # sub = 'IDU003V002'
    # skeleton_folder = os.path.join(measure_dir,'IDU003V002','RealSense', 'Skeletons')
    # skeleton_list = os.listdir(skeleton_folder)
    # skeleton_list = sorted(skeleton_list, key=lambda x: int(x.split('_')[1]))
    # pepper_log_file = os.path.join(measure_dir,sub,'Log_Files',f'{sub}_pepper.txt')

    # timestamps_stop = extract_pepper_timestamps(pepper_log_file)

    # start = timestamps_stop[0] - 1000 
    # end_time = timestamps_stop[-1] + 1000

    # print(timestamps_stop[0], start)

    # print(skeleton_list)
    # create_clip('IDU002V005')