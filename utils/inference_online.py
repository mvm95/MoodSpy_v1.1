import os
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import cv2

def get_task_table(tag):
    log_folder = os.path.join('Measures', tag, 'Log_files')
    pepper_log_file = os.path.join(log_folder, f'{tag}_pepper.txt')
    task_dict = {
        1 : 'Stop',
        0 : 'Movement'
    }
    tasks=[]
    text = ''
    with open(pepper_log_file, 'r') as f:
        for line in f.readlines():
            if line.startswith('Movement'):
                text += line
                tasks.append(0)
            if line.startswith('Stop'):
                text += line
                tasks.append(1)

    timestamps_and_tasks = re.findall(r'(?:Movement|Stop).*?(\d{13}) - ?(\d{13})', text)
    table = []
    for idx, t in enumerate(tasks):
        table.append((task_dict[t], timestamps_and_tasks[idx][0], timestamps_and_tasks[idx][1]))
    return table

def get_video_timestamps(tag, video):
    log_folder = os.path.join('Measures', tag, 'Log_files')
    connection_log_file = os.path.join(log_folder, 'Connection_log.txt')
    timestamps = []
    with open(connection_log_file, 'r') as f:
        for line in f.readlines():
            if video in line:
                timestamps.append(int(line.split(', ')[1].split('\n')[0]))
    return timestamps

def load_data(t1, t2, tag, video):
    acc_path = os.path.join('Measures', tag, video, 'BioHarness', 'ACC.csv')
    acc_data = pd.read_csv(acc_path, header=None)
    acc_signal = []
    acc_baseline = pd.read_csv(os.path.join('Measures', tag, 'Baseline', 'Bioharness', 'ACC.csv'))
    acc_baseline = acc_baseline.iloc[:, :20*3].stack().to_numpy()
    acc_baseline =acc_baseline.reshape(3, -1, order = 'F')
    for idx in range(acc_data.shape[0]):
        data = acc_data.iloc[idx, :].to_numpy()
        if (data[-1] >= int(t1) and not data[-1] >= int(t2)) and (data[-1] <= int(t2) and not data[-1] <= int(t1)):
            dt = int((acc_data.iloc[idx+1, -1] - acc_data.iloc[idx, -1])/20)
            t_vector = [t for t in range(data[-1] + dt, acc_data.iloc[idx+1,-1], dt)]
            for j in range(len(t_vector)):
                if int(t1) <= t_vector[j] <= int(t2):
                    acc_signal.append([data[3*j], data[3*j+1], data[3*j+2], t_vector[j]])
    acc_signal = np.asanyarray(acc_signal)
    acc_signal[:,0] = acc_signal[:,0]  - np.mean(acc_baseline, axis=1)[0]
    acc_signal[:,1] = acc_signal[:,1]  - np.mean(acc_baseline, axis=1)[1]
    acc_signal[:,2] = acc_signal[:,2]  - np.mean(acc_baseline, axis=1)[2]
    skeleton_path = os.path.join('Measures', tag, video, 'RealSense', 'Skeletons')
    skeleton_list = [skl for skl in os.listdir(skeleton_path) if int(t1) <= int(skl.split('_')[-1].split('.')[0]) <= int(t2)]
    skeleton_list = sorted(skeleton_list, key = lambda x: int(x.split('_')[-1].split('.')[0]))
    skeleton_timestamps = [int(x.split('_')[-1].split('.')[0]) for x in skeleton_list] 
    # teoretical sampling rate: rs = 30 hz, acc bio = 50 hz
    t1, t2 = int(t1), int(t2)
    stride = int((1/50)*1000)
    T = 250
    start_t = t1
    end_t = start_t + T
    features_list = []
    while start_t >= t1 and end_t <= t2:
        try:
            skeletons = [skel for skel, timestamp in zip(skeleton_list, skeleton_timestamps) if start_t <= timestamp <= end_t]
            skl_features = extract_features_skeletons([os.path.join(skeleton_path, sk) for sk in skeletons])
            accelerations = acc_signal[(acc_signal[:, 3] >= start_t) & (acc_signal[:, 3] <= end_t), :3]
            acc_features = extract_features_acc(accelerations)
            features_list.append(np.concatenate((acc_features, skl_features), axis = 0))
        except:
            pass
        start_t = start_t + stride
        end_t = start_t + T
    return features_list

JOINT_LIST = [num for num in range(11,33)]
def extract_features_skeletons(skeletons, velocity = True, only_velocity = False, threshold = 0.5):
    skeleton_data = np.zeros((3,len(JOINT_LIST), len(skeletons)))
    for idx, skl in enumerate(skeletons):
        df = pd.read_csv(skl)
        df = np.asanyarray(df.iloc[JOINT_LIST,1:4]).T
        skeleton_data[:,:,idx] = df
    features = []
    if not only_velocity:
        mean = np.mean(skeleton_data, axis=2).flatten()
        std = np.std(skeleton_data, axis=2).flatten()
        max_s = np.max(skeleton_data, axis=2).flatten()
        min_s = np.min(skeleton_data, axis=2).flatten()
        rmse = np.sqrt(np.mean(skeleton_data ** 2, axis=2)).flatten()
        features.extend([mean, std, max_s, min_s, rmse])
    if velocity:
        skeleton_data = np.diff(skeleton_data, axis=2)
        mean = np.mean(skeleton_data, axis=2).flatten()
        std = np.std(skeleton_data, axis=2).flatten()
        max_s = np.max(skeleton_data, axis=2).flatten()
        min_s = np.min(skeleton_data, axis=2).flatten()
        rmse = np.sqrt(np.mean(skeleton_data ** 2, axis=2)).flatten()
        features.extend([mean, std, max_s, min_s, rmse])
    return np.array(np.concatenate(features))

def extract_features_acc(data):
    data = data.T
    mean = np.mean(data, axis=1).flatten()
    std = np.std(data, axis=1).flatten()
    max = np.max(data, axis=1).flatten()
    min = np.min(data, axis=1).flatten()
    rmse = np.sqrt(np.mean(data ** 2, axis=1)).flatten()
    features=[mean, std, max, min, rmse]
    return np.array(np.concatenate(features))


ARMS_JOINT=[11, 12, 15, 16] #RS, LS, RW, LW
def load_angles_arms(t1 , t2, tag, video):
    t1, t2 = int(t1), int(t2)
    skeleton_path = os.path.join('Measures', tag, video, 'RealSense', 'Skeletons')
    skeleton_list = [skl for skl in os.listdir(skeleton_path) if t1 <= int(skl.split('_')[-1].split('.')[0]) <= t2]
    skeleton_list = sorted(skeleton_list, key = lambda x: int(x.split('_')[-1].split('.')[0]))
    skeleton_timestamps = [int(x.split('_')[-1].split('.')[0]) for x in skeleton_list] 
    angles = []
    t0 = skeleton_timestamps[0]
    j=0
    for skl in skeleton_list:
        df = pd.read_csv(os.path.join(skeleton_path, skl), header=None)
        df = np.asanyarray(df.iloc[ARMS_JOINT, [1,2, 3]]).astype(np.float32)
        image = np.zeros((480,640,3)).astype(np.uint8)
        for idx in range(4):
            # if not idx == 0:
            #     df[idx,:] = project_to_plane(df[0,:], df[idx,:])
            row = df[idx,:]
            x = int(row[0]* 640)
            y = int(row[1]* 480)
            cv2.circle(image, (x, y), radius=5, color=(0,0,255), thickness=-1)
            cv2.putText(image, f'{row[0]:.3f}, {row[1]:.3f}, {row[2]:.3f}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        for edge in [(0,1), (0,2), (1,3)]:
            p1 = edge[0]
            p2 = edge[1]
            row1 = df[p1,:]
            row2 = df[p2, :]
            x1 = int(row1[0]*640)
            y1 = int(row1[1]*480)
            x2 = int(row2[0]*640)
            y2 = int(row2[1]*480)
            cv2.line(image, (x1,y1), (x2,y2), color = (255,0,0))
        # cv2.line(image, (int(df[2,0]*640), int(df[2,1]*480)), (int(df[3,0]*640), int(df[3,1]*480)), color = (0,255,0))
        cv2.putText(image, f'{(skeleton_timestamps[j] - t0)/1000} s', (320, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow('aaa', image)
        cv2.waitKey(250)
        v01 = np.array([ df[1][1] - df[0][1], df[1][0] - df[0][0]])
        v02 =-np.array([ df[2][1] - df[0][1], df[2][0] - df[0][0]])
        v13 = np.array([ df[3][1] - df[1][1], df[3][0] - df[1][0]])
        a1 =  np.arctan2(np.linalg.det([v01,v02]), np.dot(v01,v02))
        a2 = np.arctan2(np.linalg.det([v01,v13]), np.dot(v01,v13))
        a1 = np.arctan2(v02[0], v02[1],)
        a2 = np.arctan2(v13[0], v13[1])
        ang = [np.degrees(a1), np.degrees(a2)]
        print(ang)
        angles.append(ang)
        j+=1
    cv2.destroyAllWindows()
    return angles


def inference_online(tag, video):
    table = get_task_table(tag)
    timestamps_video = get_video_timestamps(tag, video)
    video_tasks = []
    for t in table:
        if int(t[1]) >= timestamps_video[0] and int(t[2]) <= timestamps_video[1]:
            video_tasks.append(t)
    with open('rf_3c.pkl', 'rb') as f:
        model = pickle.load(f)
    scaler = StandardScaler()
    for task in video_tasks:
        print(task)
        if task[0] == 'Stop':
            continue
            x = load_data(task[1], task[2], tag, video)
            x = scaler.fit_transform(x)
            y = model.predict(x)
            with open(os.path.join('Measures', tag, 'inference.txt'), 'a') as f:
                f.write(f'{video}')
                for pred in y:
                   f.write(f',{pred}')
                f.write('\n')
        elif task[0] == 'Movement':
            x = load_angles_arms(task[1], task[2], tag, video)
            x = np.asanyarray(x)
            # import matplotlib.pyplot as plt
            # fig = plt.figure()
            # plt.plot(x[0,:])
            # plt.plot(x[1,:])
            # plt.show()
           # IMPLEMENT CLASSIFIER ERROR

def main():
    tag = 'marco'
    for video in  os.listdir(f'Measures/{tag}'):
        if 'V' in video:
            print(video)
            # try:
            inference_online(tag, video)
            # except:
            #     pass

if __name__ == '__main__':
    main()