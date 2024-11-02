import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

DATA_PATH = 'd:/Marco'
subject = 'IDU019'
video = 'V017'
fs = 250
nsample = 63

def getBaselineSignal(subject):
    ecg_path = os.path.join(DATA_PATH, subject, 'Stop', 'Baseline', 'BioHarness', 'ECG.csv')
    ecg_df = pd.read_csv(ecg_path, header=None)
    signal = ecg_df.iloc[:, :nsample*3]
    signal = signal.stack().to_numpy()
    return signal

def butterBandpassFilter(signal):
    fNy = fs/2
    lc = .5/fNy
    hc = 50/fNy
    cut = 2
    # b,a = butter(5, [lc,hc], btype = 'band')
    b,a = butter(5, cut/fNy, btype='high' )
    y = filtfilt(b,a,signal)
    return y

def extract_actions_from_path(subject):
    log_path = os.path.join(DATA_PATH, subject)
    log_path = os.path.join(log_path, 'Stop', 'Log_files', f'{subject}_pepper.txt')
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

def createTimeVector(df):
    ts0 = df.iloc[0, -1] - nsample * 1000 / fs
    n_row = df.shape[0]
    t = []
    for idx in range(n_row):
        ts = df.iloc[idx, -1]
        step = (ts - ts0) / (nsample - 1)  # Ensure exactly nsample points
        t.extend(np.linspace(ts0, ts, nsample, endpoint=False))
        ts0 = ts
    return t

def plotEcg():
    ecg_path = os.path.join(DATA_PATH, subject, 'Stop', subject+video, 'BioHarness', 'ECG.csv')
    ecg_df = pd.read_csv(ecg_path, header = None)
    signal = ecg_df.iloc[:, :nsample]
    signal = signal.stack().to_numpy()
    t = np.asanyarray(createTimeVector(ecg_df))
    actionTstamps = extract_actions_from_path(subject)
    verticalLines = []
    for action in actionTstamps:
        ts1, ts2, label = action
        print(action, t[0], t[-1])
        if t[0] <= ts1 <= t[-1]:   
            index = np.argmin(np.abs(t-ts1))
            verticalLines.append((index, label))
            index_yellow = np.argmin(np.abs(t-ts2))
            verticalLines.append((index_yellow, 3))
    start_idx = verticalLines[0][0] - 50
    end_idx = verticalLines[-1][0] + 50 
    start_idx = 0
    end_idx = len(signal) - 1
    t = t[start_idx:end_idx]
    # signal = signal[start_idx:end_idx]

    fig = plt.figure()
    plt.plot(t-t[0], signal[start_idx:end_idx], label = 'ECG Signal')
    for index, label in verticalLines:
        if label == 0:
            color = 'green'
        elif label == 3:
            color = 'yellow'
        else:
            color = 'red'
        plt.vlines(t[index-start_idx]-t[0], ymin =min(signal), ymax = max(signal), colors=color)
    plt.grid()
    plt.legend()
    plt.title('ECG signal')
    fig.show()
    filtered_signal = butterBandpassFilter(signal)
    fig = plt.figure()
    plt.plot(t-t[0], filtered_signal[start_idx:end_idx], label = 'Filtered ECG Signal')
    for index, label in verticalLines:
        if label == 0:
            color = 'green'
        elif label == 3:
            color = 'yellow'
        else:
            color = 'red'
        plt.vlines(t[index-start_idx]-t[0], ymin =min(filtered_signal), ymax = max(filtered_signal), colors=color)
    plt.grid()
    plt.legend()
    plt.title('Filtered ECG signal')
    plt.show()
plotEcg()