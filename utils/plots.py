import os
from visualizedata_bioharness import extract_pepper_timestamps, obtain_t_axis
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from rs_utils import post_processing_extraction



measure_dir= 'Measures'

def plot_synchronized_signals(subject):
    # # pepper_log_file = os.path.join(measure_dir,subject,'Log_Files',f'{subject}_pepper.txt')
    # print(pepper_log_file)
    # # timestamps = extract_pepper_timestamps(pepper_log_file)

    fig, axs = plt.subplots(7,1, figsize = (10, 28), sharex=True)
    
    #ECG
    fc = 250
    nsample = 63
    data = pd.read_csv(os.path.join(measure_dir,subject, 'BioHarness', 'ECG.csv'), header = None) 
    t, tv =obtain_t_axis(data, nsample, fc)
    ts0 = data.iloc[0,-1] 

    signal = data.iloc[:, :nsample]
    signal = signal.stack().to_numpy()
    # ts = (timestamps-ts0)/1000
    # if 'IDU002' in subject or subject == 'IDU003V001':
    #     for idx, timestamp in enumerate(ts):
    #         if idx % 2 == 1:
    #             timestamp = timestamp + 1000


    # # ts = [np.abs(tv - time).argmin() for time in timestamps]
    # plot(ts)
    # print(ts)
    # ts = [np.abs(t - time).argmin() for time in ts]
    # print(ts)

    axs[0].plot(t, signal, label = 'ECG')
    axs[0].set_title('ECG')
    axs[0].legend()
    # # for idx, line in enumerate(ts):
    # #     color = 'r' if idx % 2 == 0 else 'g'
    # #     axs[0].axvline(x=t[line], color=color, linestyle='--')

    #RR
    fc = 18
    nsample = 17
    data = pd.read_csv(os.path.join(measure_dir,subject, 'BioHarness',  'RR.csv'), header = None) 
    t, tv=obtain_t_axis(data, nsample, fc)
    ts0 = data.iloc[0,-1]
    signal = data.iloc[:, :nsample]
    signal = signal.stack().to_numpy()
    # ts = (timestamps-ts0)/1000
    # # ts = [np.abs(tv - time).argmin() for time in timestamps]

    # ts = [np.abs(t - time).argmin() for time in ts]

    axs[1].plot(t, signal, label = 'RR')
    axs[1].set_title('RR')
    axs[1].legend()
    # # for idx, line in enumerate(ts):
    # #     color = 'r' if idx % 2 == 0 else 'g'
    # #     axs[1].axvline(x=t[line], color=color, linestyle='--')

    #BREATH
    fc = 18
    nsample = 18
    data = pd.read_csv(os.path.join(measure_dir, subject,'BioHarness', 'BR.csv'), header = None) 
    t, tv=obtain_t_axis(data, nsample, fc)
    ts0 = data.iloc[0,-1]
    signal = data.iloc[:, :nsample]
    signal = signal.stack().to_numpy()
    # ts = (timestamps-ts0)/1000
    # # ts = [np.abs(tv - time).argmin() for time in timestamps]

    # ts = [np.abs(t - time).argmin() for time in ts]

    axs[2].plot(t, signal, label = 'BREATHING')
    axs[2].set_title('BREATHING')
    axs[2].legend()
    # # for idx, line in enumerate(ts):
    # #     color = 'r' if idx % 2 == 0 else 'g'
    # #     axs[2].axvline(x=t[line], color=color, linestyle='--')

    #ACCELERATION
    fc = 50
    nsample = 20
    data = pd.read_csv(os.path.join(measure_dir,subject, 'BioHarness', 'ACC.csv'), header = None) 
    t, tv=obtain_t_axis(data, nsample, fc)
    ts0 = data.iloc[0,-1]
    signal = data.iloc[:, :nsample*3]
    signal = signal.stack().to_numpy()
    sig_x = signal[::3]
    sig_y = signal[1::3]
    sig_z = signal[2::3]
    # ts = (timestamps-ts0)/1000
    # # ts = [np.abs(tv - time).argmin() for time in timestamps]

    # ts = [np.abs(t - time).argmin() for time in ts]

    axs[3].plot(t, sig_x, label = 'ACCELERATION X')
    axs[3].set_title('ACCELERATION X')
    axs[3].legend()
    # # for idx, line in enumerate(ts):
    # #     color = 'r' if idx % 2 == 0 else 'g'
    # #     axs[3].axvline(x=t[line], color=color, linestyle='--')

    axs[4].plot(t, sig_y, label = 'ACCELERATION Y')
    axs[4].set_title('ACCELERATION Y')
    axs[4].legend()
    # # for idx, line in enumerate(ts):
    # #     color = 'r' if idx % 2 == 0 else 'g'
    # #     axs[4].axvline(x=t[line], color=color, linestyle='--')

    axs[5].plot(t, sig_z, label = 'ACCELERATION Z')
    axs[5].set_title('ACCELERATION Z')
    axs[5].legend()
    # # for idx, line in enumerate(ts):
    # #     color = 'r' if idx % 2 == 0 else 'g'
    # #     axs[5].axvline(x=t[line], color=color, linestyle='--')

    signal = (sig_x**2 + sig_y**2 + sig_z**2)**.5
    axs[6].plot(t, signal, label = 'ACCELERATION MODULE')
    axs[6].set_title('ACCELERATION MODULE')
    axs[6].legend()
    # # for idx, line in enumerate(ts):
    # #     color = 'r' if idx % 2 == 0 else 'g'
    # #     axs[6].axvline(x=t[line], color=color, linestyle='--')

    plt.tight_layout()
    # plt.show()
    subject = os.path.normpath(subject).replace("\\", "_")
    plt.savefig(os.path.join(measure_dir, 'Plots', f'{subject}.png'))
    plt.close()

def modulo_acc(subject):
    pepper_log_file = os.path.join(measure_dir,'Log_files',subject,f'{subject}_pepper.txt')
    # # timestamps = extract_pepper_timestamps(pepper_log_file)
    fc = 50
    nsample = 20
    data = pd.read_csv(os.path.join(measure_dir, 'BioHarness', subject, 'ACC.csv'), header = None) 
    t=obtain_t_axis(data, nsample, fc)
    ts0 = data.iloc[0,-1]
    signal = data.iloc[:, :nsample*3]
    signal = signal.stack().to_numpy()
    sig_x = signal[::3]
    sig_x = sig_x -sig_x.mean()
    sig_y = signal[1::3]
    sig_y = sig_y -sig_y.mean()
    sig_z = signal[2::3]
    sig_z = sig_z -sig_z.mean()

    signal = (sig_x**2 + sig_y**2 + sig_z**2)**.5
    # ts = (timestamps-ts0)/1000

    plt.plot(t, signal)
    # for idx, line in enumerate(ts):
    #     color = 'r' if idx % 2 == 0 else 'g'
    #     plt.vlines(xt=[line], ymin=min(signal), ymax=max(signal), color=color, linestyle='--')

    plt.show()


from tqdm import tqdm
def main():
    subjects = os.listdir('Measures')
    print(subjects)
    for sub in subjects:
        if sub == 'Plots':
             continue
        # if 'IDU002' in sub or 'IDU003V001' in sub:
        print(sub)
        for video in os.listdir(os.path.join('Measures', sub)):
                video_id = os.path.join(sub,video)
                print(video)
                try:
                    if not os.path.exists(f'Measures/Plots/'):
                        os.makedirs(f'Measures/Plots/')
                    video_id = os.path.join(sub,video)
                    # print(video_id)
                    plot_synchronized_signals(video_id)
                except Exception as e:
                    print(e)
                



if __name__ == '__main__':
    main()
    # plot_synchronized_signals('IDU001V005')
    # print('ok')
    # subjects = os.listdir('/home/marco/Desktop/Codes/MoodSpy/Measures/Bioharness')
    # for sub in tqdm(subjects):
    #     if 'IDU' in sub:
    #         print(sub)
    #         post_processing_extraction(sub)
    #                     # break
    # sub = 'IDU003V001'
    # plot_synchronized_signals(sub)
