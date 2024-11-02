import os
import numpy as np
from sklearn.cluster import KMeans
test_subject = ['IDU006', 'IDU001']
T=500

# Definisci la cartella contenente gli array numpy
file_features = {}

subjects = os.listdir('/home/marco/Desktop/Codes/MoodSpy/data/')
for sub in subjects:
    if sub[0:6] in test_subject:
        continue
    save_data_path = f'/home/marco/Desktop/Codes/MoodSpy/data/{sub}/ACCELEROMETER/{T}'

    # Elenco dei file nella cartella
    file_list = os.listdir(save_data_path)

    # Inizializza un dizionario per contenere i nomi dei file e le caratteristiche
    # print(len(file_list))
    # Per ogni file nella cartella
    for data in file_list:
        # Costruisci il percorso completo del file
        data_path = os.path.join(save_data_path, data)

        # Carica l'array numpy dal file
        this_data = np.load(data_path)
        # print(this_data)

        # Calcola le caratteristiche (media e deviazione standard lungo gli assi)
        features = np.concatenate((np.mean(this_data, axis=1), np.std(this_data, axis=1)))
        # features = np.mean(this_data, axis=1)
        # print(features)
        if any( np.isnan(features)):
            print(data, sub)
            continue

        # Salva le caratteristiche e il nome del file nel dizionario
        file_features[data] = {"features": features, "name": data}

    # Estrai i nomi dei file e le caratteristiche dal dizionario
# print(file_features)
feature_names = [info["name"] for info in file_features.values()]
features_array = np.array([info["features"] for info in file_features.values()])

# Addestra il modello KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(features_array)

# Ottieni le etichette di cluster assegnate a ciascun puntoIDU001V005
labels = kmeans.labels_
print(labels)
print(len(labels))
    # Puoi anche stampare le etichette insieme ai nomi dei file
import pandas as pd
test = 'IDU001V003'

save_data_path = f'/home/marco/Desktop/Codes/MoodSpy/data/{test}/ACCELEROMETER/{T}'

# Elenco dei file nella cartella
file_list = os.listdir(save_data_path)

# Inizializza un dizionario per contenere i nomi dei file e le caratteristiche
# print(len(file_list))
# Per ogni file nella cartella
file_features = {}
for data in file_list:
    # Costruisci il percorso completo del file
    data_path = os.path.join(save_data_path, data)

    # Carica l'array numpy dal file
    this_data = np.load(data_path)

    # Calcola le caratteristiche (media e deviazione standard lungo gli assi)
    features = np.concatenate((np.mean(this_data, axis=1), np.std(this_data, axis=1)))
    # features = np.mean(this_data, axis=1)
    if any( np.isnan(features)):
        print(data, test, features, this_data)
        continue

    # Salva le caratteristiche e il nome del file nel dizionario
    file_features[data] = {"features": features, "name": data}


features_array = np.array([info["features"] for info in file_features.values()])
feature_names = [info["name"] for info in file_features.values()]

labels = kmeans.predict(features_array)

fc = 50
nsample = 20
data = pd.read_csv(os.path.join('/home/marco/Desktop/Codes/MoodSpy/Measures', test, 'Bioharness', 'ACC.csv'), header = None) 
from utils.visualizedata_bioharness import obtain_t_axis
_, t=obtain_t_axis(data, nsample, fc)

import matplotlib.pyplot as plt
#  data_col_idx = list((data.iloc[0] == 0) | (data.iloc[0] > 2000)).index(True) - 1
data = data.loc[:, :nsample*3-1]
data = data.stack().to_numpy()
acc_x = data[::3]
acc_y = data[1::3]
acc_z = data[2::3]

fig = plt.figure(figsize=(12,6))
plt.plot(t, acc_x)
# fig, axs = plt.subplots(3, figsize=(8,6), sharex=False)
# t=np.arange(len(acc_x))/(fc)
# axs[0].plot(t,acc_x)
# axs[0].set_title('Acceleration - x')
# axs[0].set_xlabel('Time (s)')
# axs[0].set_ylabel('x-acceleration')
# axs[1].plot(t,acc_y)
# axs[1].set_title('Acceleration - y')
# axs[1].set_xlabel('Time (s)')
# axs[1].set_ylabel('y-acceleration')
# axs[2].plot(t,acc_z)
# axs[2].set_title('Acceleration - z')
# axs[2].set_xlabel('Time (s)')
# axs[2].set_ylabel('z-acceleration')
plt.tight_layout()

    # from visualizedata_bioharness import plot_acc, obtain_t_axis
    # import matplotlib.pyplot as plt
    # from tqdm import tqdm

    # plot_acc('IDU001V005', data_dir = '/home/marco/Desktop/Codes/MoodSpy/Measures', time_axis='ts', save = False)
    # # _, ts = obtain_t_axis()
    # # plt.plot(ts, np.arange(len(ts)))
for i, feature_name in enumerate(feature_names):
    label = labels[i]
    timestamps = [int(feature_name.split('_')[1]),int(feature_name.split('_')[3].split('.')[0])]
    # print(timestamps)
    color = 'red' if label == 0 else 'green'
    # for t in tqdm(range(timestamps[0], timestamps[1])):
        # Change the background color for the segment from 200 to 300
    plt.axvspan(timestamps[0], timestamps[1], facecolor=color, alpha=0.3)

    # Change the background color for the segment from 2000 to 2100
    # plt.axvspan(2000, 2100, facecolor='cyan', alpha=0.3, label='Background (2000-2100)')

        # plt.scatter(t, 0, c=color, label=f'Label: {label}\nTimestamps: {timestamps[0]} - {timestamps[1]}')

    # print(f"File: {feature_name}, Label: {labels[i]}")
    # print(len(feature_name))
plt.show()