import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import re

# data_dir_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Measures") 
data_dir_script = 'Measures'
def extract_pepper_timestamps(filename):
    timestamp_pattern = re.compile(r'\d+')

    timestamps = []

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('Stop task from timestamps'):
                # Extract timestamps using regular expression
                timestamps.extend(map(int, re.findall(timestamp_pattern, line)))

    if 'IDU002' in filename or 'IDU003V001' in filename:
        # print(timestamps)
        for idx, ts in enumerate(timestamps):
            if idx % 2 == 1:
                timestamps[idx] = ts + 1000
        # print(timestamps)
    
    # if 'IDU003V001' in filename:
    #     timestamps = timestamps[12:15]
        


    return timestamps


def obtain_t_axis(data, nsample, fs):
    n_row = data.shape[0]
    start = 0
    t = np.arange(nsample)/fs
    ts = data.iloc[:,-1]
    first_ts = ts.iloc[0] - (nsample*1000/fs)
    ts_vec = np.arange(nsample)*1000/fs + first_ts
    for i in range(1, n_row):
        start = t[-1] + 1/fs
        first_ts = ts_vec[-1] + (1000/fs)
        fs = nsample/((ts.iloc[i] - ts.iloc[i-1])/1000)
        tv = np.arange(nsample)/fs + start
        tsv = np.arange(nsample)*1000/fs + first_ts
        t = np.concatenate((t, tv), axis = 0)
        ts_vec = np.concatenate((ts_vec, tsv), axis=0)
    return t, ts_vec

def plot_ecg(tag_text, save = True, data_dir = data_dir_script):
    fc = 250
    nsample = 63

    data = pd.read_csv(os.path.join(data_dir, tag_text, 'Bioharness', 'ECG.csv'), header = None) 
    t, _ =obtain_t_axis(data, nsample, fc)
    # print(t)
    T = (data.iloc[-1,-1] - data.iloc[0,-1])/1000
  

 #   data_col_idx = list((data.iloc[0] == 0) | (data.iloc[0] > 2000)).index(True) - 1
    data = data.loc[:, :nsample-1]
    data = data.stack().to_numpy()
    t_per_plot = 10
    # sample_tot = len(data)
    n_plots = int(np.ceil(T/ t_per_plot))

    if n_plots>1:
        fig, axs = plt.subplots(n_plots, figsize=(8,2*n_plots), sharex = True)

        for idx in range(n_plots):
            # start_idx = int(idx*t_per_plot*fc)
            start_idx = np.abs(t - idx*t_per_plot).argmin()
            # end_idx = int((idx+1)*t_per_plot*fc)
            end_idx = np.abs(t - t_per_plot*(idx+1)).argmin()
            # t = np.arange(len(data[start_idx:end_idx])) / fc 
            # axs[idx].plot(t, data[start_idx:end_idx])
            axs[idx].plot(t[start_idx:end_idx]-t[start_idx], data[start_idx:end_idx])
            axs[idx].set_title(f'Segment {idx + 1}')
            axs[idx].set_xlabel('Time (s)')
            axs[idx].set_ylabel('ECG Signal')
    else:
        # t = np.arange(len(data)) / fc
        plt.figure(figsize=(10,4))
        plt.plot(t,data)
        plt.title('ECG Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('ECG Signal')

    
    plt.tight_layout()
    if save: 
        plt.savefig(os.path.join(data_dir,tag_text, 'Bioharness','Segnale ECG.png'))  # Sostituisci 'nome_del_tuo_file' con il nome desiderato
    else:
        plt.show()

def plot_rr(tag_text, save = True, data_dir = data_dir_script):
    fc = 18
    nsample = 17
    data = pd.read_csv(os.path.join(data_dir, tag_text, 'Bioharness', 'RR.csv'), header = None) 
    t, _ =obtain_t_axis(data, nsample, fc)
    # T = (data.iloc[-1,-1] - data.iloc[0,-1])/1000
  #  data_col_idx = list((data.iloc[0] == 0) | (data.iloc[0] > 2000)).index(True) - 1
    data = data.loc[:, :nsample-1]
    data = data.stack().to_numpy()

    # t = np.arange(len(data)) / fc
    plt.figure(figsize=(10,4))
    plt.plot(t,data)
    plt.title('RR Signal')
    plt.xlabel('Sample')
    plt.ylabel('RR Signal')

    if save:
        plt.savefig(os.path.join(data_dir,tag_text, 'Bioharness','Segnale RR.png'))  # Sostituisci 'nome_del_tuo_file' con il nome desiderato
    else:
        plt.show()

def plot_breath(tag_text, save = True, data_dir = data_dir_script):
    fc = 18
    nsample = 18
    data = pd.read_csv(os.path.join(data_dir, tag_text, 'Bioharness', 'BR.csv'), header = None) 
    t, _ =obtain_t_axis(data, nsample, fc)
    T = (data.iloc[-1,-1] - data.iloc[0,-1])/1000
    # data_col_idx = list((data.iloc[0] == 0) | (data.iloc[0] > 2000)).index(True) - 1
    data = data.loc[:, :nsample-1]
    data = data.stack().to_numpy()

    # T = len(data)/fc
    t_per_plot = 60
    n_plots = int(np.ceil(T/ t_per_plot))

    if n_plots>1:
        fig, axs = plt.subplots(n_plots, figsize=(8,2*n_plots), sharex=True)

        for idx in range(n_plots):
            # start_idx = int(idx*t_per_plot*fc)
            start_idx = np.abs(t - idx*t_per_plot).argmin()
            # end_idx = int((idx+1)*t_per_plot*fc)
            end_idx = np.abs(t - t_per_plot*(idx+1)).argmin()
            # t = np.arange(len(data[start_idx:end_idx])) / fc 
            # axs[idx].plot(t, data[start_idx:end_idx])
            axs[idx].plot(t[start_idx:end_idx]-t[start_idx], data[start_idx:end_idx])
            axs[idx].set_title(f'Segment {idx + 1}')
            axs[idx].set_xlabel('Time (s)')
            axs[idx].set_ylabel('Breathing Waveform Signal')
        
        plt.tight_layout()
    else:
        # t = np.arange(len(data)) / fc
        plt.figure(figsize=(10,4))
        plt.plot(t,data)
        plt.title('BR Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('BR Signal')


    if save: 
        plt.savefig(os.path.join(data_dir,tag_text, 'Bioharness', 'Segnale Breathing Waveform.png'))  # Sostituisci 'nome_del_tuo_file' con il nome desiderato
    else:
        plt.show()

def plot_acc(tag_text, save=True, data_dir = data_dir_script, time_axis = 's'):
    fc = 50
    nsample = 20
    data = pd.read_csv(os.path.join(data_dir, tag_text, 'Bioharness', 'ACC.csv'), header = None) 
    if time_axis == 's':
        t, _=obtain_t_axis(data, nsample, fc)
    else:
        _, t=obtain_t_axis(data, nsample, fc)

    
  #  data_col_idx = list((data.iloc[0] == 0) | (data.iloc[0] > 2000)).index(True) - 1
    data = data.loc[:, :nsample*3-1]
    data = data.stack().to_numpy()
    acc_x = data[::3]
    acc_y = data[1::3]
    acc_z = data[2::3]
    fig, axs = plt.subplots(3, figsize=(8,6), sharex=False)
    # t=np.arange(len(acc_x))/(fc)
    axs[0].plot(t,acc_x)
    axs[0].set_title('Acceleration - x')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('x-acceleration')
    axs[1].plot(t,acc_y)
    axs[1].set_title('Acceleration - y')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('y-acceleration')
    axs[2].plot(t,acc_z)
    axs[2].set_title('Acceleration - z')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('z-acceleration')
    plt.tight_layout()
    if save: 
        plt.savefig(os.path.join(data_dir,tag_text, 'Bioharness', 'Acceleration.png'))  # Sostituisci 'nome_del_tuo_file' con il nome desiderato
    else:
        plt.show()




def plot_data(tag_text, data_dir = data_dir_script):
    plot_breath(tag_text,data_dir= data_dir )
    plot_ecg(tag_text, data_dir= data_dir)
    plot_rr(tag_text, data_dir= data_dir )
    plot_acc(tag_text, data_dir= data_dir )
    plt.close()

if __name__ == '__main__':
    tag_text = 'prova_corsa'
    plot_data(tag_text)
    