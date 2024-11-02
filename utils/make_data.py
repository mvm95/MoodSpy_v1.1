import os
from visualizedata_bioharness import extract_pepper_timestamps, obtain_t_axis
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import math

def is_between(pair, ts):
    return pair[0] <= ts <= pair[1]

DATA_PATH = 'Previous Project/data'
MEASURE_PATH = 'Previous Project/Measures'

# subject = 'IDU001V005'
T = 125 #ms
stride = int(T/2)
test_subject = ['IDU006', 'IDU001']

labels = pd.read_csv('Previous Project/Etichettatura.csv', header = None)

def create_data_acc(subject, T, stride = None):
    try:
        this_label = labels[labels[0] == subject].values.tolist()[0]
        this_label = this_label[1:]
    # print(this_label)
        this_label = [ast.literal_eval(item) for item in this_label if not (isinstance(item, float) and np.isnan(item))]
    except IndexError:
        return
    if stride == None:
        stride = int(T/2)
    save_data_path = os.path.join(DATA_PATH, subject, 'ACCELEROMETER',str(T))
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)

  # print(this_label)
    label_file = os.path.join(save_data_path, 'labels.txt')
    pepper_log_file = os.path.join(MEASURE_PATH,subject,'Log_Files',f'{subject}_pepper.txt')
    # print(pepper_log_file)
    timestamps = extract_pepper_timestamps(pepper_log_file)
    timestamps = np.asanyarray(timestamps)
    time_table = timestamps.reshape(-1,2)

    ts_tables = [np.arange(time[0], time[1]) for time in time_table]
    idx_keep= []
    start_times = np.asanyarray(this_label)[:,0]
    for idx, ts in enumerate(ts_tables):
        for st in start_times:
            if st >= ts[0] and st <= ts[-1]:
                # print(st, ts[0], ts[-1])
                idx_keep.append(idx)
    # print(idx_keep)
    idx_keep = sorted(list(set(idx_keep)) )
    # print(idx_keep)
    # indices_to_keep = [i for i, ts_table in enumerate(ts_tables) if any((start_times >= ts_table[0]) & (start_times <= ts_table[-1]))]
    time_table = [time_table[idx] for idx in idx_keep]
    # print(time_table)
    fc = 50
    nsample = 20
    data = pd.read_csv(os.path.join(MEASURE_PATH,subject, 'Bioharness', 'ACC.csv'), header = None) 
    t, tv=obtain_t_axis(data, nsample, fc)
    
    signal = data.iloc[:, :nsample*3]
    signal = signal.stack().to_numpy()
    signal = signal.reshape(3, -1, order='F')
    # signal = signal - np.mean(signal, axis=1, keepdims=True)
    # print(signal.shape)
    # print(timestamps)
    ts_label = np.zeros_like(tv)
    for l in this_label:
        # l = ast.literal_eval(l)
        s = np.abs(tv-l[0]).argmin()
        e = np.abs(tv-l[1]).argmin()
        ts_label[s:e] = l[2]

    start_t = timestamps[0]
    end_t = start_t + T
    # count=0
    # i=0
    while end_t <= timestamps[-1]:
        idx_start = np.abs(tv - start_t).argmin()
        idx_end = np.abs(tv - end_t).argmin()
        # print(idx_start, idx_end)
        if any(is_between(pair, start_t) for pair in time_table) and any(is_between(pair, end_t) for pair in time_table) and any(is_between(pair, act[0]) for pair in time_table for act in this_label):
            this_data = signal[:,idx_start:idx_end]
            if not this_data.size == 0:
                # print('no data')
            # print(this_data.shape)
                data_name = f'from_{start_t}_to_{end_t}.npy'
                unique_values, counts = np.unique(ts_label[idx_start:idx_end], return_counts=True)
                mode_index = np.argmax(counts)
                label = int(unique_values[mode_index])
                np.save(os.path.join(save_data_path,data_name), this_data)
                with open( label_file, 'a') as f:
                    f.write(f'{data_name},{label}\n')
                # count +=1
        start_t = start_t + stride
        end_t = start_t + T
        # i += 1
        # print(i,count, (start_t - data.iloc[0,-1])/1000)

def create_data_skl(subject, T, stride = None):
    try:
        this_label = labels[labels[0] == subject].values.tolist()[0]
        this_label = this_label[1:]
    # print(this_label)
        this_label = [ast.literal_eval(item) for item in this_label if not (isinstance(item, float) and np.isnan(item))]
    except IndexError:
        return
    if stride == None:
        stride = int(T/2)
    save_data_path = os.path.join(DATA_PATH, subject, 'SKELETONS',str(T))
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)

    label_file = os.path.join(save_data_path, 'labels.txt')
    pepper_log_file = os.path.join(MEASURE_PATH,subject,'Log_Files',f'{subject}_pepper.txt')
    # print(pepper_log_file)
    timestamps = extract_pepper_timestamps(pepper_log_file)
    timestamps = np.asanyarray(timestamps)
    time_table = timestamps.reshape(-1,2)

    ts_tables = [np.arange(time[0], time[1]) for time in time_table]
    idx_keep= []
    start_times = np.asanyarray(this_label)[:,0]
    for idx, ts in enumerate(ts_tables):
        for st in start_times:
            if st >= ts[0] and st <= ts[-1]:
                # print(st, ts[0], ts[-1])
                idx_keep.append(idx)
    # print(idx_keep)
    idx_keep = sorted(list(set(idx_keep)) )
    # print(idx_keep)
    # indices_to_keep = [i for i, ts_table in enumerate(ts_tables) if any((start_times >= ts_table[0]) & (start_times <= ts_table[-1]))]
    time_table = [time_table[idx] for idx in idx_keep]

    data_folder = os.path.join(MEASURE_PATH, subject, 'RealSense', 'Skeletons')
    data_list = os.listdir(data_folder)
    data_list = sorted(data_list, key = lambda x: int(x.split('_')[-1].split('.')[0]))
    timestamp_skl_list = [data.split('_')[-1].split('.')[0] for data in data_list]
    timestamp_skl_list = np.asanyarray(timestamp_skl_list).astype(np.float64)
    # print(timestamp_skl_list)
    first_ts = int(int(timestamp_skl_list[0])/1000)
    last_ts = int(int(timestamp_skl_list[-1])/1000)

    tv = np.arange(first_ts, last_ts, 1000/30) #teoretical frame rate
    ts_label = np.zeros_like(tv)
    for l in this_label:
        # l = ast.literal_eval(l)
        s = np.abs(tv-l[0]).argmin()
        e = np.abs(tv-l[1]).argmin()
        ts_label[s:e] = l[2]

    joint_list = [num for num in range(11,33)]
    #perchè ho sbagliato
    start_ts = first_ts
    end_ts = start_ts + T
    print(start_ts, end_ts, last_ts)
    print(time_table)
    while end_ts <= last_ts:
        if any(is_between(pair, start_ts) for pair in time_table) and any(is_between(pair, end_ts) for pair in time_table) and any(is_between(pair, act[0]) for pair in time_table for act in this_label):
            idx_start = np.abs(tv - start_ts).argmin()
            idx_end = np.abs(tv - end_ts).argmin()
            timestamp_vector = tv[idx_start: idx_end]
            skeleton_data = np.zeros((4,len(joint_list), int(np.ceil(T*30/1000))))
            # assert(len(skeleton_data) == int(np.ceil(T*30/1000)))
            for idx, ts in enumerate(timestamp_vector):
                if np.abs(timestamp_skl_list - ts).min() <= 1000/30:
                    # print(ts,timestamp_skl_list[np.abs(timestamp_skl_list - ts).argmin()])
                    skeleton_idx = np.abs(timestamp_skl_list-ts).argmin()
                    df = pd.read_csv(os.path.join(data_folder,data_list[skeleton_idx]))
                    df = np.asanyarray(df.iloc[joint_list,1:]).T
                    skeleton_data[:,:,idx] = df
                if idx == 0:
                    start_t =int( timestamp_skl_list[np.abs(timestamp_skl_list-ts).argmin()])
                if idx == len(timestamp_vector) - 1:
                    end_t = int(timestamp_skl_list[np.abs(timestamp_skl_list-ts).argmin()])

            if not np.sum(skeleton_data) == 0:
                # print(start_ts)
                data_name = f'from_{start_t}_to_{end_t}.npy'
                print('ok')
                unique_values, counts = np.unique(ts_label[idx_start:idx_end], return_counts=True)
                mode_index = np.argmax(counts)
                label = int(unique_values[mode_index])
                np.save(os.path.join(save_data_path,data_name), skeleton_data)
                with open( label_file, 'a') as f:
                    f.write(f'{data_name},{label}\n')
        start_ts = start_ts + stride
        end_ts = start_ts + T
from tqdm import tqdm
def main():
    subjects = os.listdir(MEASURE_PATH)
    # T = 125
    for T in [125,250,500]:
        for sub in sorted(subjects):
            print(sub, T)
            if sub[:6] in test_subject:
                stride = 1
            else:
                stride = int(T/2)
            try:
                # pass
                create_data_skl(sub, T, stride = stride)
                create_data_acc(sub, T, stride=stride)
            # except Exception as e:
            except FileNotFoundError:
                pass
            #     print(sub, e)


if __name__ == '__main__':
    main()

# fig =plt.figure()
# for idx in range(3):
#     plt.plot(tv, signal[idx,:] - signal[idx].mean())
# plt.show()
# for idx, timestamp in enumerate(tv):
#     if any(is_between(pair, timestamp) for pair in timestamps) and any(is_between(pair, timestamp) for pair in timestamps):
