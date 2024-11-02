import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
import numpy as np
from scipy.signal import butter, filtfilt
import pywt

DATA_PATH = 'd:/Marco'
subject = 'IDU014'
video = 'V001'
trial = 'Movement'
fs = 50
nsample = 20
# SUBJECT_LIST = sorted(os.listdir(DATA_PATH))

def wavelets(data):
    wavelet = 'haar'
    uselevels = 7
    mode = 'zero'
    levels = (np.floor(np.log2(data.shape[0]))).astype(int)
    omit = levels - uselevels
    coeffs = pywt.wavedec(data, wavelet, level = levels)
    A = pywt.waverec(coeffs[:-omit] + [None] * omit, wavelet, mode=mode)
    return A

def band_pass_filter(data, order=5):
    data = data.astype(np.float64) - data.mean()
    nyquist = 0.5 * fs
    low = .1 / nyquist
    high = 6.0 / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def createTimeVector(acc_df):
    ts0 = acc_df.iloc[0, -1] - nsample * 1000 / fs
    n_row = acc_df.shape[0]
    t = []
    for idx in range(n_row):
        ts = acc_df.iloc[idx, -1]
        step = (ts - ts0) / (nsample - 1)  # Ensure exactly nsample points
        t.extend(np.linspace(ts0, ts, nsample, endpoint=False))
        ts0 = ts
    return t

def extract_actions_from_path(subject):
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

def getBaselineSignal(subject):
    acc_path = os.path.join(DATA_PATH, subject, trial, 'Baseline', 'BioHarness', 'ACC.csv')
    acc_df = pd.read_csv(acc_path, header=None)
    signal = acc_df.iloc[:, :nsample*3]
    signal = signal.stack().to_numpy()
    acc_x = signal[::3]
    acc_y = signal[1::3]
    acc_z = signal[2::3]
    return acc_x, acc_y, acc_z

def plotVector(ax, ay, az):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.animation as animation
    fig = plt.figure()
    n_points = len(ax)
    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection='3d')

    # Set the plot limits
    ax3d.set_xlim([0, 3])
    ax3d.set_ylim([0, 3])
    ax3d.set_zlim([0, 3])

    # Labels
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')

    # Plot initialization function
    def update(num):
        ax3d.cla()  # Clear the axis
        ax3d.quiver(0, 0, 0, ax[num], ay[num], az[num])
        ax3d.set_xlim([0, 3])
        ax3d.set_ylim([0, 3])
        ax3d.set_zlim([0, 3])
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Z')
        ax3d.set_title(f'Frame {num + 1}')
        return fig,

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=n_points, repeat=False)
    plt.show()

def plotAccComponents(signal, title):
    acc_x = signal[::3]
    acc_y = signal[1::3]
    acc_z = signal[2::3]
    fig = plt.figure()
    plt.plot(acc_x, label = 'X-component (gravity)')
    plt.plot(acc_y,  label = 'Y-component (lateral)')
    plt.plot(acc_z, label = 'Z-component (sagittal)')
    plt.title(title)
    plt.grid()
    plt.legend()
    fig.show()

def figureModuleAcc(file, frame):
    acc_df = pd.read_csv(file, header=None)
    signal = acc_df.iloc[:, :nsample*3]
    signal = signal.stack().to_numpy().astype(np.float64) #- 511
    t = np.asanyarray(createTimeVector(acc_df))
    actionTstamps = extract_actions_from_path(subject)
    verticalLines = []
    for action in actionTstamps:
        ts1, ts2, label = action
        if t[0] <= ts1 <= t[-1]:
            index = np.argmin(np.abs(t-ts1))
            verticalLines.append((index, label))
            index_yellow = np.argmin(np.abs(t-ts2))
            verticalLines.append((index_yellow, 3))
    start_idx = verticalLines[0][0] - 100
    end_idx = verticalLines[-1][0] + 100
    start_idx = 0
    end_idx = len(t) - 1
    # t = t/1000
    # t = t[start_idx:end_idx]
    # signal = signal[start_idx:end_idx]
    signal = signal 
    # print(x_mean.shape)
    acc_x = signal[::3]
    acc_x -= acc_x[start_idx:end_idx].mean()
    print(acc_x.shape)
    acc_y = signal[1::3]
    acc_y -= acc_y[start_idx:end_idx].mean()
    acc_z = signal[2::3]
    acc_z -= acc_z[start_idx:end_idx].mean()
    # acc_x = acc_x[start_idx:end_idx]
    # acc_y = acc_y[start_idx:end_idx]
    # acc_z = acc_z[start_idx:end_idx]
    acc = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    figure = plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t,acc)
    for index, label in verticalLines:
        if label == 0:
            color = 'green'
        elif label == 3:
            color = 'yellow'
        else:
            color = 'red'
        plt.vlines(t[index], ymin =min(acc), ymax = max(acc), colors=color)
    plt.vlines(t[frame] , ymin =min(acc), ymax = max(acc), colors='black', linestyles='--')
    plt.vlines()
    plt.title('Acceleration')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.grid(True)
    plt.legend()
    return figure

def plotAcc():
    acc_path = os.path.join(DATA_PATH, subject, trial, 'Baseline', 'BioHarness', 'ACC.csv')
    log_path = os.path.join(DATA_PATH, subject, trial, 'Log_files', subject+'_pepper.txt')
    # acc_path = r'C:\Users\marco\OneDrive - unifi.it\Codes\Sensor\ACC.csv'
    acc_df = pd.read_csv(acc_path, header=None)
    ts0 = acc_df.iloc[0,-1] - nsample*1000/fs
    signal = acc_df.iloc[:, :nsample*3]
    signal = signal.stack().to_numpy().astype(np.float64)# - 511
    # signal = signal/(531-512)
    # signal = signal*9.81
    # signal = np.vstack((signal[::3], signal[1::3], signal[2::3])).T
    z_mean = np.round(signal[2::3].mean())
    y_mean = np.round(signal[1::3].mean())
    # signal = signal-z_mean
    # signal = signal - np.round(np.mean(signal))
    x_mean = np.round(np.abs(signal[::3].mean()))
    print(x_mean, y_mean, z_mean)
    # signal = signal/x_mean
    # signal = signal*9.81

    # signal = (signal - 2131)/83
    plotAccComponents(signal, 'baseline')
    acc_path = os.path.join(DATA_PATH, subject, trial, subject+video, 'BioHarness', 'ACC.csv')
    acc_df = pd.read_csv(acc_path, header=None)
    ts0 = acc_df.iloc[0,-1] - nsample*1000/fs
    signal = acc_df.iloc[:, :nsample*3]
    signal = signal.stack().to_numpy().astype(np.float64) #- 511
    # signal = signal/(531-512)
    # signal = np.vstack([signal[::3], signal[1::3], signal[2::3]])
    # signal = signal - np.round(np.mean(signal))

    # signal = (signal-z_mean)/x_mean
    # signal = signal*9.81
    # signal = (signal - 2131)/83
    # acc_x = (signal[::3] - 512)/19.456
    # acc_y = (signal[1::3] - 512)/19.456
    # acc_z = (signal[2::3] - 512)/19.456
    # acc_x = (signal[::3] - 2131)/160
    # acc_y = (signal[1::3] - 2131)/160
    # acc_z = (signal[2::3] - 2131)/160

    # plotAccComponents(signal, 'raw_signal')
    # acc_x -= acc_x.mean()
    # acc_y -= acc_y.mean()
    # acc_z -= acc_z.mean()
    # baseline = getBaselineSignal(subject)
    # acc_x -= baseline[0].mean()
    # acc_y -= baseline[1].mean()
    # acc_z -= baseline[2].mean()
    # acc_x = band_pass_filter(acc_x)
    # acc_y = band_pass_filter(acc_y)
    # acc_z = band_pass_filter(acc_z)
    # acc_x = wavelets(acc_x)
    # acc_y = wavelets(acc_y)
    # acc_z = wavelets(acc_z)
    # plotVector(acc_x, acc_y, acc_z)
    # acc_z = acc_y
    # acc_z -= acc_z.mean()
    t = np.asanyarray(createTimeVector(acc_df))
    actionTstamps = extract_actions_from_path(subject)
    verticalLines = []
    for action in actionTstamps:
        ts1, ts2, label = action
        if t[0] <= ts1 <= t[-1]:
            index = np.argmin(np.abs(t-ts1))
            verticalLines.append((index, label))
            index_yellow = np.argmin(np.abs(t-ts2))
            verticalLines.append((index_yellow, 3))

    # print(dt)
    # vel_z = vel_x = vel_y = np.zeros_like(acc_z)
    # print(acc_z.mean())
    # acc_z = acc_z - acc_z.mean()
    # vel_z = cumtrapz(acc_z, dx = dt, initial=0)
    # vel_y = cumtrapz(acc_y, dx = dt, initial=0)
    # vel_x = cumtrapz(acc_x, dx = dt, initial=0)
    # vel_z = cumtrapz(acc_z, dx = 0.02, initial=0)
    # vel_y = cumtrapz(acc_y, dx = 0.02, initial=0)
    # vel_x = cumtrapz(acc_x, dx = 0.02, initial=0)
    start_idx = verticalLines[0][0] - 50
    end_idx = verticalLines[-1][0] + 50
    start_idx = 0
    end_idx = len(t) - 1
    # t = t/1000
    t = t[start_idx:end_idx]
    # signal = signal[start_idx:end_idx]
    signal = signal 
    print(x_mean.shape)
    acc_x = signal[::3]- x_mean
    print(acc_x.shape)
    acc_y = signal[1::3] - y_mean
    acc_z = signal[2::3] - z_mean
    acc_x = acc_x[start_idx:end_idx]
    acc_y = acc_y[start_idx:end_idx]
    acc_z = acc_z[start_idx:end_idx]
    # acc_x -= acc_x.mean()
    # acc_y -= acc_y.mean()
    # acc_z -= acc_z.mean()
    # acc_x = band_pass_filter(acc_x)
    # acc_y = band_pass_filter(acc_y)
    # acc_z = band_pass_filter(acc_z)
    acc = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    acc = acc_z
    # acc = acc_z -z_mean
    # acc = acc - acc.mean()
    fig = plt.figure()
    plt.plot(t/1000, acc_x, label = 'X-component (gravity)')
    plt.plot(t/1000, acc_y,  label = 'Y-component (lateral)')
    plt.plot(t/1000, acc_z, label = 'Z-component (sagittal)')
    # for index, label in verticalLines:
    #     if label == 0:
    #         color = 'green'
    #     elif label == 3:
    #         color = 'yellow'
    #     else:
    #         color = 'red'
    #     plt.vlines(t[index-start_idx]-t[0], ymin =min(acc_x), ymax = max(acc_x), colors=color)
    plt.grid()
    plt.legend()
    plt.title('Acceleration components')
    fig.show()
    dt = np.diff(t)/1000
    # acc = acc[start_idx:end_idx]
    # vel = cumtrapz(acc, dx=dt, initial=0)
    dacc = np.diff(acc)
    jerk = np.zeros_like(acc)
    jerk[0] = (acc[1] - acc[0])/(t[1]-t[0])
    jerk[-1] = (acc[-1] - acc[-1])/(t[-1]-t[-2])
    for idx in range(1, len(acc)-1):
        jerk[idx] = (acc[idx+1] - acc[idx-1])/(t[idx+1]-t[idx-1])
    vel_x, vel_y, vel_z = np.zeros_like(acc_x), np.zeros_like(acc_x), np.zeros_like(acc_x)
    vel = np.zeros_like(acc)
    for i in range(1, len(dt)+1):
        # print(acc_z[i])
        # vel[i] = vel[i-1] + acc[i]*dt[i-1]
        vel_z[i] = vel_z[i-1] + acc_z[i] * dt[i-1]  #acc_z[i]
        vel_x[i] = vel_x[i-1] + acc_x[i] * dt[i-1]  #acc_x[i]
        vel_y[i] = vel_y[i-1] + acc_y[i] * dt[i-1]  #acc_y[i]

    vel = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
    vel = vel_z
    # vel_x = band_pass_filter(vel_x)
    # vel_y = band_pass_filter(vel_y)
    # vel_z = band_pass_filter(vel_z)
    jerk[0] = (vel[1] - vel[0])/(t[1]-t[0])
    jerk[-1] = (vel[-1] - vel[-1])/(t[-1]-t[-2])
    for idx in range(1, len(acc)-1):
        jerk[idx] = (vel[idx+1] - vel[idx-1])/(t[idx+1]-t[idx-1])



    # vel_z = vel_z - 875/1000*np.asanyarray([float(time) for time in t-t[0]])
    # acc = acc_z
    # vel = vel_z
    fig_acc = plt.figure()
    plt.subplot(3,1,1)
    # plt.plot(acc_x, label='Acc - X')
    # plt.plot(acc_y, label='Acc - Y')
    # plt.plot(acc_z, label='Acc - Z')
    plt.plot(t-t[0],acc)
    # plt.plot(t-t[0], acc_x, label='Acc - X')
    # plt.plot(t-t[0], acc_y, label='Acc - Y')
    # plt.plot(t-t[0], acc_z, label='Acc - Z')
    for index, label in verticalLines:
        if label == 0:
            color = 'green'
        elif label == 3:
            color = 'yellow'
        else:
            color = 'red'
        plt.vlines(t[index-start_idx]-t[0], ymin =min(acc), ymax = max(acc), colors=color)
        # plt.vlines(t[index-start_idx]-t[0], colors=color, ymin =min([acc_x.min(), acc_y.min(), acc_z.min()]), ymax = max([acc_x.max(), acc_y.max(), acc_z.max()]))
    plt.title('Acceleration')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t-t[0],vel)
    # plt.plot(t-t[0], vel_x, label='Velocity - X')
    # plt.plot(t-t[0], vel_y, label='Velocity - Y')
    # plt.plot(t-t[0], vel_z, label='Velocity - Z')
    # plt.plot(vel_x, label='Velocity - X')
    # plt.plot(vel_y, label='Velocity - Y')
    # plt.plot( vel_z, label='Velocity - Z')
    for index, label in verticalLines:
        if label == 0:
            color = 'green'
        elif label == 3:
            color = 'yellow'
        else:
            color = 'red'
        plt.vlines(t[index-start_idx]-t[0], ymin =min(vel), ymax = max(vel), colors=color)
        # plt.vlines(t[index-start_idx]-t[0], ymin =min([vel_x.min(), vel_y.min(), vel_z.min()]), ymax = max([vel_x.max(), vel_y.max(), vel_z.max()]), colors=color)
    plt.title('Velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True)
    plt.legend()

    mov_x, mov_y, mov_z = np.zeros_like(acc_x), np.zeros_like(acc_x), np.zeros_like(acc_x)
    mov = np.zeros_like(acc)
    for i in range(1, len(dt)+1):
        # print(acc_z[i])
        # vel[i] = vel[i-1] + acc[i]*dt[i-1]
        mov_z[i] = mov_z[i-1] + vel_z[i] * dt[i-1]  #acc_z[i]
        mov_x[i] = mov_x[i-1] + mov_x[i] * dt[i-1]  #acc_x[i]
        mov_y[i] = mov_y[i-1] + mov_y[i] * dt[i-1]  #acc_y[i]
    mov = np.sqrt(mov_x **2 + mov_y ** 2 + mov_z**2)
    mov = mov_z
    plt.subplot(3, 1, 3)
    plt.plot(t-t[0],mov)
    # plt.plot(t-t[0], vel_x, label='Velocity - X')
    # plt.plot(t-t[0], vel_y, label='Velocity - Y')
    # plt.plot(t-t[0], vel_z, label='Velocity - Z')
    # plt.plot(vel_x, label='Velocity - X')
    # plt.plot(vel_y, label='Velocity - Y')
    # plt.plot( vel_z, label='Velocity - Z')
    for index, label in verticalLines:
        if label == 0:
            color = 'green'
        elif label == 3:
            color = 'yellow'
        else:
            color = 'red'
        plt.vlines(t[index-start_idx]-t[0], ymin =min(jerk), ymax = max(jerk), colors=color)
        # plt.vlines(t[index-start_idx]-t[0], ymin =min([vel_x.min(), vel_y.min(), vel_z.min()]), ymax = max([vel_x.max(), vel_y.max(), vel_z.max()]), colors=color)
    plt.title('Movement')
    plt.xlabel('Time (s)')
    # plt.ylabel('Jerk (m/s^3)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

plotAcc()