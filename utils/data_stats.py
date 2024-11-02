import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from visualizedata_bioharness import extract_pepper_timestamps, obtain_t_axis
import ast
from itertools import groupby
# import time

DATA_PATH = '/home/marco/Desktop/Codes/MoodSpy/data'
MEASURE_PATH = '/home/marco/Desktop/Codes/MoodSpy/Measures'

def create_action_csv():
    subjects = os.listdir(MEASURE_PATH)
    labels = pd.read_csv('/home/marco/Desktop/Codes/MoodSpy/Etichettatura.csv', header = None)

    df = []
    columns = ['ID', 'VIDEO', 'ACTION', 'LENGTH[ms]']
    # subjects = ['IDU003V001']
    for sub in sorted(subjects):
        print(sub)
        try:
            this_label = labels[labels[0] == sub].values.tolist()[0]
            this_label = this_label[1:]
            # print(this_label)
            this_label = [ast.literal_eval(item) for item in this_label if not (isinstance(item, float) and np.isnan(item))]
        # print(this_label)
        except IndexError:
            continue
        pepper_log_file = os.path.join(MEASURE_PATH,sub,'Log_Files',f'{sub}_pepper.txt')
        timestamps = extract_pepper_timestamps(pepper_log_file)
        timestamps = np.asanyarray(timestamps)
        time_table = timestamps.reshape(-1,2)  
        # print(time_table)
        this_label = np.asanyarray(this_label)
        time_table = np.asanyarray(time_table)
        # print(this_label)
        start_times = this_label[:,0]
        # print(this_label.shape, start_times.shape)


        ts_tables = [np.arange(time[0], time[1]) for time in time_table]
        print(len(ts_tables))
        idx_keep= []
        for idx, ts in enumerate(ts_tables):
            for st in start_times:
                if st >= ts[0] and st <= ts[-1]:
                    # print(st, ts[0], ts[-1])
                    idx_keep.append(idx)
        # print(idx_keep)
        idx_keep = sorted(list(set(idx_keep)) )
        # print(idx_keep)
        # indices_to_keep = [i for i, ts_table in enumerate(ts_tables) if any((start_times >= ts_table[0]) & (start_times <= ts_table[-1]))]
        ts_tables = [ts_tables[idx] for idx in idx_keep]
        vector = np.hstack(ts_tables)
        time_array = np.zeros_like(vector)
        for start, end, label in this_label:
            # print(start, end)
            time_array[(vector >= start) & (vector <= end)] = label

        results = []
        for key, group in groupby(time_array):
            results.append((key, len(list(group))))

        for result in results:
            if result[1] > 33:
                data = {
                    'ID' : sub[:6],
                    'VIDEO' : sub,
                    'ACTION' : result[0],
                    'LENGTH[ms]' : result[1]
                }
                df.append(data)

    df = pd.DataFrame(df, columns=columns)
    df.to_csv(os.path.join('/home/marco/Desktop/Codes/MoodSpy/data_stats', 'actions.csv'), index = False)

def create_data_csv(path = DATA_PATH):
    
    for T in ['125', '250', '500']:
        # T = '125'
        for sensor in ['ACCELEROMETER', 'SKELETONS']:
            subjects = os.listdir(DATA_PATH)
            df = []
            columns = ['ID', 'VIDEO', 'num_tot_data', 'num_0', 'num_1', 'num_2']
            for sub in sorted(subjects):
                filename = os.path.join(DATA_PATH, sub, sensor, T, 'labels.txt')
                if not os.path.exists(filename):
                    continue
                total = 0
                c0 = 0
                c1 = 0
                c2 = 0
                with open(filename, 'r') as f:
                    for line in f:
                        total += 1
                        _, label = line.split(',')
                        label = int(label)
                        if label == 0:
                            c0 += 1
                        elif label == 1:
                            c1 += 1
                        elif label == 2:
                            c2 += 1
                data = {
                    'ID' : sub[:6],
                    'VIDEO' : sub,
                    'num_tot_data' : total,
                    'num_0' : c0,
                    'num_1' : c1,
                    'num_2' : c2
                }
                df.append(data)
            df = pd.DataFrame(df, columns=columns)
            df.to_csv(os.path.join('/home/marco/Desktop/Codes/MoodSpy/data_stats', f'data_{sensor.lower()}_{T}.csv'), index = False)




if __name__ == '__main__':
    # create_data_csv()
    create_action_csv()