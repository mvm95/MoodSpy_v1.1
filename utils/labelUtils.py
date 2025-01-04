import ast
import math
import cv2
from matplotlib import pyplot as plt
import rosbag
import os
import pandas as pd
import numpy as np
# import mediapipe as mp
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
# pose = mp_pose.Pose(min_detection_confidence = 0.1, min_tracking_confidence =0.1) 

def get_skeletons(frame, path, timestamp):
    connections = [(11, 13), (13, 15), (15, 19), (15, 17), (15, 21), (17, 19), (11, 12), (12, 14),
                    (14, 16), (16, 18), (16, 22), (16, 20), (18, 20), (11, 23), (23, 25), (25, 27),
                    (27, 29), (27, 31), (29, 31), (12, 24), (23, 24), (24, 26), (26, 28), (28, 30),
                    (28, 32), (30, 32) ]
    # timestamp = int(timestamp/10)
    path = os.path.dirname(path)
    path = os.path.join(path,'Skeletons')
    skeleton_list = os.listdir(path)
    skeleton_list = sorted(skeleton_list, key = lambda x: int(x.split('_')[1]))
    skeleton_ts = [int(skel.split('_')[-1][:-4]) for skel in skeleton_list]
    skel_index = min(range(len(skeleton_ts)), key = lambda x: abs(skeleton_ts[x] - timestamp))
    skel_file = skeleton_list[skel_index]
    skel = pd.read_csv(os.path.join(path, skel_file))
    h, w, c = frame.shape
    for _, row in skel.iterrows():
        x = int(row['x-norm']* w)
        y = int(row['y-norm']* h)
        cv2.circle(frame, (x, y), radius=2, color=(0,0,255), thickness=-1)  
    for edge in connections:
        p1 = edge[0]
        p2 = edge[1]
        row1 = skel.iloc[p1]
        row2 = skel.iloc[p2]
        x1 = int(row1['x-norm']*w)
        y1 = int(row1['y-norm']*h)
        x2 = int(row2['x-norm']*w)
        y2 = int(row2['y-norm']*h)
        cv2.line(frame, (x1,y1), (x2,y2), color = (255,255,255))     
    return frame
    # frame.flags.writeable = False
    # results = pose.process(frame)
    # frame.flags.writeable = True
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # if results.pose_landmarks:
    #     mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    # return frame

def get_topic_table(reader):
    info = reader.get_type_and_topic_info()

    topic_tuple = info.topics.values()
    topics = info.topics.keys()

    message_types = [t1.msg_type for t1 in topic_tuple]
    n_messages = [t1.message_count for t1 in topic_tuple]
    frequency = [t1.frequency for t1 in topic_tuple]

    topic_table = pd.DataFrame(list(zip(topics, message_types, n_messages, frequency)), columns=['Topics', 'Types', 'Message Count', 'Frequency'])
    return topic_table

def extract_actions_from_path(bagfile):
    bagfile, filename = os.path.split(bagfile)
    id_user = filename[:6]
    log_path = os.path.dirname(bagfile)
    log_path = os.path.dirname(log_path)
    log_path = os.path.join(log_path, 'Log_files', f'{id_user}_pepper.txt')
    actions_timestamps = []
    with open(log_path, 'r') as f:
        for line in f:
            if 'task from timestamps' in line:
                text = line.split(',')[0]
                timestamps = text.split(': ')[1]
                timestamps = timestamps.split(' - ')
                ts1 = int(timestamps[0])
                ts2 = int(timestamps[1])
                if 'Go' in line:
                    label = 0
                if 'Stop' in line:
                    label = 1
                if 'Movement' in line:
                    label = 2
                actions_timestamps.append((ts1, ts2, label))
    return actions_timestamps

def is_a_task(ts, actions_list):
    for t1,t2,label in actions_list:
        if t1 <= ts <= t2:
            return (t1, t2, label)
    return None


def extract_video_from_path(bagfile):
    reader = rosbag.Bag(bagfile)
    topic_table = get_topic_table(reader)
    topic_table = topic_table[(topic_table['Types'].str.contains('sensor_msgs')) & (topic_table['Message Count'] > 1)]
    topics = list(topic_table['Topics'])
    actions_list = extract_actions_from_path(bagfile)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    font_color = (255, 255, 255) 
    frames = []
    timestamps=[]
    tstart = actions_list[0][0] - 1000
    for topic, msg, t in reader.read_messages(topics=topics):
        if 'Color' in topic:
            ts = msg.header.stamp.to_nsec()
            ts = int(ts/1000000)
            if ts < tstart:
                continue
            color_data = msg.data
            w = msg.width
            h = msg.height
            color_data = np.frombuffer(color_data, dtype = np.uint8).reshape((h,w,3)).copy()
            label = -1
            color = (255,255,255)
            isTask = is_a_task(ts, actions_list)
            text = str(ts)
            if isTask:
                t1,t2,label = isTask
                if label==0:
                    color = (0,255,0)
                if label == 1:
                    color = (0,0,255)
                if label == 2:
                    color = (0, 165, 255)
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_position = ((color_data.shape[1] - text_size[0]) // 2, 40)  # Adjust the Y coordinate as 
        #    color_data = get_skeletons(color_data)
            cv2.putText(color_data, text, text_position, font, font_scale, (0,0,0), font_thickness+1)
            cv2.putText(color_data, text, text_position, font, font_scale, color, font_thickness)   
            color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
            frames.append(color_data)
            timestamps.append(ts)
    return frames, timestamps

def update_label_file(label, first_ts, last_ts, video_name):
    filename = 'Etichettatura_v2.xlsx'
    ncolumns=20
    try:
        df = pd.read_excel(filename)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Video'])
        for i in range(ncolumns):
            df[f'column_{i+1}'] = None
    if video_name not in df['Video'].values:
        df.loc[len(df)] = [video_name] + [None] * (ncolumns)
    idx = df.index[df['Video'] == video_name][0]
    tmp_list = df.iloc[idx, 1:].tolist()
    for j in range(len(tmp_list)):
        if not type(tmp_list[j]) == str:
            if tmp_list[j] == None or np.isnan(tmp_list[j]):
               break
    item_to_insert=(label, first_ts, last_ts)
    df.iloc[idx, j+1] = str(item_to_insert)
    df = df.sort_values(by='Video')
    for i in range(df.shape[0]):
        row = df.iloc[i, 1:].tolist()
        tmp_row = []
        for j, item in enumerate(row):
            if type(item) == str and item != 'None':
                tmp_row.append(item)
            elif type(item) == np.float64 and not np.isnan(item):
                tmp_row.append(item)
        tmp_row = sorted(tmp_row, key= lambda x: ast.literal_eval(x)[1] if x else None)
        for j, action in enumerate(tmp_row):
            df.iloc[i, j+1] = action
    df.to_excel(filename, index=False)

def get_number_of_task(path):
    file = 'Etichettatura_v2.xlsx'
    if not os.path.exists(file):
        return 0
    df = pd.read_excel(file)
    idU = path[path.find('IDU'):path.find('IDU')+6]
    df = df[df['Video'].str.startswith(idU)]
    count = 0
    for index, row in df.iterrows():
        for col in df.columns:
            if col == 'Video':
                continue
            try:
                action = ast.literal_eval(str(row[col]))[0]
                # print(action)
                if action == 1:
                    count +=1
            except Exception as e:
                # print(e)
                continue
    # print(count)
    return count

def createTimeVector(acc_df):
    nsample = 20
    fs = 50
    ts0 = acc_df.iloc[0, -1] - nsample * 1000 / fs
    n_row = acc_df.shape[0]
    t = []
    for idx in range(n_row):
        ts = acc_df.iloc[idx, -1]
        step = (ts - ts0) / (nsample - 1)  # Ensure exactly nsample points
        t.extend(np.linspace(ts0, ts, nsample, endpoint=False))
        ts0 = ts
    return t

def extract_actions_from_path(path):
    folders = path.split('/')
    DATA_PATH = folders[0] + '/' + folders[1]
    subject = folders[2]
    trial = folders[3]
    log_path = os.path.join(DATA_PATH, subject)
    log_path = os.path.join(log_path, trial, 'Log_files', f'{subject}_pepper.txt')
    actions_timestamps = []
    with open(log_path, 'r') as f:
        for line in f:
            if 'task from timestamps' in line:
                text = line.split(',')[0]
                timestamps = text.split(': ')[1]
                timestamps = timestamps.split(' - ')
                ts1 = int(timestamps[0])
                ts2 = int(timestamps[1])
                if 'Go' in line:
                    label = 0
                if trial in line:
                    label = 1
                if 'Movement' in line:
                    label = 2
                actions_timestamps.append((ts1, ts2, label))
    return actions_timestamps

def figureModuleAcc(file, timestamp, ax):
    acc_df = pd.read_csv(file, header=None)
    nsample = 20

    signal = acc_df.iloc[:, :nsample*3]
    signal = signal.stack().to_numpy().astype(np.float64)
    t = np.asanyarray(createTimeVector(acc_df))
    actionTstamps = extract_actions_from_path(file)
    verticalLines = []
    
    for action in actionTstamps:
        ts1, ts2, label = action
        if t[0] <= ts1 <= t[-1]:
            index = np.argmin(np.abs(t-ts1))
            verticalLines.append((index, label))
            index_yellow = np.argmin(np.abs(t-ts2))
            verticalLines.append((index_yellow, 3))
    # print(t, timestamp)
    ts_index =  np.argmin(np.abs(t-timestamp))
    start_idx = max(ts_index - 150,0)
    end_idx = min(ts_index + 150, len(t)-1)
    # print(ts_index, start_idx, end_idx)
    acc_x = signal[::3][start_idx:end_idx]
    acc_x -= acc_x.mean()
    acc_y = signal[1::3][start_idx:end_idx]
    acc_y -= acc_y.mean()
    acc_z = signal[2::3][start_idx:end_idx]
    acc_z -= acc_z.mean()
    acc = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)*9.81 - 9.81
    # Plot on provided axis
    t_plot = t[start_idx:end_idx]
    # t_plot = t
    ax.clear()
    ax.plot(t_plot , acc)
    
    for index, label in verticalLines:
        if start_idx < index < end_idx:
            color = 'green' if label == 0 else 'yellow' if label == 3 else 'red'
            ax.vlines(t[index], ymin=min(acc), ymax=max(acc), colors=color)

    ax.vlines(t[ts_index] , ymin=min(acc), ymax=max(acc), colors='black', linestyles='--')
    ax.set_title('Acceleration')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (m/s^2)')
    ax.grid(True)
    ax.figure.set_size_inches(12, 6) 



def acceleration_plot(path, frame, ax):
    acc_path = os.path.join(os.path.dirname(os.path.dirname(path)), 'BioHarness', 'ACC.csv')
    figureModuleAcc(acc_path, frame, ax)

                
if __name__ == '__main__':
    subject = 'IDU002'
    group = 'Stop'
    # update_label_file(1, -4, 2, 'IDU010V010')
    # file = "D:\Marco\IDU001\Movement\IDU001V001\BioHarness\ACC.csv".replace('\\','/')
    # timestamp = 1726670682498
    # figure = plt.Figure(figsize=(12, 6), dpi=100)
    # ax = figure.add_subplot(111)
    # figureModuleAcc(file, timestamp, ax)
    # plt.show()
    path = f"D:/Marco/{subject}/{group}/{subject}V001/RealSense/{subject}.bag"
    path = path.replace('\\', '/')
    actions = extract_actions_from_path(path)
    action_to_consider = []
    for action in actions:
        t1,t2,label = action
        if label > 0:
            action_to_consider.append(action)
    # print(action_to_consider)
    print(len(action_to_consider))
    count = 0
    video_path = f'D:/Marco/{subject}/{group}/'
    video_list = sorted([video for video in os.listdir(video_path) if 'V0' in video])
    bagname = f'{subject}.bag'
    video_log = os.path.join(video_path, 'Log_files', 'Connection_log.txt')
    with open(video_log, 'r' ) as f:
        lines = f.readlines()
    aa = []
    bb = []
    for idx, line in enumerate(lines):
        if line.startswith('Started'):
            if 'Baseline' in line:
                continue
            t1 = int(line.split(',')[1].strip())
            t2 = int(lines[idx+1].split(',')[1].strip())
            video = line.split(',')[0].split(' ')[-1]
            for action in action_to_consider:
                ta = int(action[0])
                if t1 < ta < t2:
                    aa.append((video, action))
                    bb.append(action)
    print(aa)
    this_bitch = None
    for action in action_to_consider:
        if not action in bb:
            this_bitch = action
            break
    t1b, t2b, labelb = this_bitch
    for idx, action in enumerate(action_to_consider):
        t01, t02, _ = action
        if t1b > t01 and t2b > t02 and t2b:
            print(idx+1)
        
    


    # from tqdm import tqdm
    # video_to_consider = []
    # for video in tqdm(video_list):
    #     bagfile = os.path.join(video_path, video, 'RealSense', bagname )
    #     reader = rosbag.Bag(bagfile)
    #     topic_table = get_topic_table(reader)
    #     topic_table = topic_table[(topic_table['Types'].str.contains('sensor_msgs')) & (topic_table['Message Count'] > 1)]
    #     topics = list(topic_table['Topics'])
    #     for topic, msg, t in reader.read_messages(topics=topics):
    #         if 'Color' in topic:
    #             ts = msg.header.stamp.to_nsec()
    #             ts = int(ts/1000000)
    #             isTask = is_a_task(ts, actions)
    #             text = str(ts)
    #             if isTask:
    #                 t1,t2,label = isTask
    #                 if label >0:
    #                     video_to_consider.append(video)
    #                     actions.remove((t1,t2,label))
    #                     count +=1
    # print(count)
    # print(video_to_consider)
    # print(len(video_to_consider))
    # print(actions)