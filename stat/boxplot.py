import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Mappatura dei task
def boxplot():
    TASK_TYPE_MAP = {0: 'GONOGO_COMPUTER', 1: 'STROOP_TEST', 2: 'T1', 3: 'T3', 4: 'T2', 5: 'T2+T1'}

    # Leggi il file CSV
    df = pd.read_csv('stat/reaction_times.csv')

    # Filtra i dati in base ai valori di TYPE_OF_TASK
    tasks = [2, 3, 4, 5]
    df_filtered = df[df['TYPE_OF_TASK'].isin(tasks)]
    print(df_filtered[(df_filtered['TYPE_OF_TASK'] == 3) & (df_filtered['REACTION_TIME'] > 5000)])
    # Controlla che la colonna REACTION_TIME sia numerica
    df_filtered['REACTION_TIME'] = pd.to_numeric(df_filtered['REACTION_TIME'], errors='coerce')

    # Rimuovi eventuali valori NaN che potrebbero essere stati generati dalla coercizione
    df_filtered = df_filtered.dropna(subset=['REACTION_TIME'])

    # Imposta lo stile di Seaborn
    sns.set(style="whitegrid")

    # Palette di colori personalizzata
    palette = sns.color_palette("Set2")

    # Crea un boxplot per ogni TYPE_OF_TASK
    for task in tasks:
        plt.figure(figsize=(12, 8))
        
        # Filtra il DataFrame per il task corrente
        df_task = df_filtered[df_filtered['TYPE_OF_TASK'] == task]
        min_reaction_time = df_task['REACTION_TIME'].min()
        max_reaction_time = df_task['REACTION_TIME'].max()

        # Imposta un margine extra per rendere il grafico più chiaro
        ylim_min = min_reaction_time - 150  # Aggiunge un margine inferiore
        ylim_max = max_reaction_time + 150  # Aggiunge un margine superiore
        
        # Crea il boxplot per il task corrente con una palette di colori
        sns.boxplot(x='SUBJECT', y='REACTION_TIME', data=df_task, palette=palette, linewidth=2, fliersize=4)
        
        # Aggiungi griglia leggera
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Usa il mapping TASK_TYPE_MAP per il titolo
        plt.title(f'Boxplot dei REACTION TIMES {TASK_TYPE_MAP[task]} per SUBJECT', fontsize=16, fontweight='bold')
        
        # Etichette degli assi
        plt.xlabel('Soggetto', fontsize=12)
        plt.ylabel('Tempo di Reazione [ms]', fontsize=12)
        plt.ylim(ylim_min, ylim_max)
        # Personalizza le etichette degli assi
        plt.xticks(rotation=90, fontsize=12)
        plt.yticks(fontsize=12)
        
        # Rimuovi i bordi superflui
        sns.despine(left=True, bottom=True)

        # Mostra il grafico con layout ottimizzato
        plt.tight_layout()
        plt.savefig(f'stat/boxplot_{TASK_TYPE_MAP[task]}.png')
def histogram():
    TASK_TYPE_MAP = {0: 'GONOGO_COMPUTER', 1: 'STROOP_TEST', 2: 'T1', 3: 'T3', 4: 'T2', 5: 'T2+T1'}

    # Leggi il file CSV
    info_file = pd.read_excel('stat\Status Reclutamento.xlsx', usecols='B,D,F', header=None)
    info_file = info_file.iloc[11:, :]
    info_file.columns = ['USER', 'TYPE', 'LABELED']
    info_file = info_file.loc[info_file['LABELED'] == 'Y', ['USER', 'TYPE']]
    movement_list = info_file[info_file['TYPE'] == 'Movement']['USER'].to_list()
    movement_list = [f'IDU{int(mov):03d}' for mov in movement_list]
    stop_list = info_file[info_file['TYPE'] == 'Stop']['USER'].to_list()
    stop_list = [f'IDU{int(stop):03d}' for stop in stop_list]
    df = pd.read_csv('stat/reaction_times.csv')
    df['REACTION_TIME'] = df['REACTION_TIME'].fillna(0)
    df['CORRECT'] = df['CORRECT'].fillna(1)
    df_stop = df[df['SUBJECT'].isin(stop_list)]
    df_movement = df[df['SUBJECT'].isin(movement_list)]
    # df = df[df['TYPE_OF_TASK'] == 1]
    # print(len(df[df['CORRECT'] == 0]))
    # print(len(df[df['REACTION_TIME'] <= 100]))
    # df = df[df['CORRECT'] != 0]
    # df = df[df['REACTION_TIME'] > 100]
    # df = df['REACTION_TIME']
    plt.figure(figsize=(10, 6))
    # sns.histplot(df, binwidth=5, kde=True, color='gray', edgecolor='black', linewidth=0.5)
    # # plt.title(f'Distribution of  (mean) - STOP CONDITION', fontweight='bold')
    # plt.xlabel('Reaction Time [ms]', fontweight='bold')
    # plt.ylabel('Density', fontweight='bold')
    # plt.grid()
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['left'].set_visible(False)
    # plt.gca().spines['bottom'].set_visible(False)
    # plt.axvline(df.mean(), color='red', linestyle='dashed', linewidth=1, label='Mean Reaction Time')
    # plt.legend()
    # plt.savefig(f'stat/histogram_stroop.png')
    for type_task in [2,3]:
        # STOP CONDITION
        stop = df_stop[df_stop['TYPE_OF_TASK'] == type_task]
        stop = stop['REACTION_TIME']
        sns.histplot(stop, binwidth=5, kde=True, color='gray', edgecolor='black')
        plt.title(f'Distribution of {TASK_TYPE_MAP[type_task]} (mean) - STOP CONDITION', fontweight='bold')
        plt.xlabel('Reaction Time [ms]', fontweight='bold')
        plt.ylabel('Density', fontweight='bold')
        plt.axvline(stop.mean(), color='red', linestyle='dashed', linewidth=1, label='Mean Reaction Time')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.grid()
        plt.legend()
        plt.savefig(f'stat/histogram_{TASK_TYPE_MAP[type_task]}_stop.png')
        plt.clf()

        # MOVEMENT CONDITION
        mov = df_movement[df_movement['TYPE_OF_TASK'] == type_task]
        mov = mov['REACTION_TIME']
        sns.histplot(mov, binwidth=5, kde=True, color='gray', edgecolor='black')
        plt.title(f'Distribution of {TASK_TYPE_MAP[type_task]} (mean) - MOVEMENT CONDITION', fontweight='bold')
        plt.xlabel('Reaction Time [ms]', fontweight='bold')
        plt.ylabel('Density', fontweight='bold')
        plt.axvline(mov.mean(), color='red', linestyle='dashed', linewidth=1, label='Mean Reaction Time')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.grid()
        plt.legend()
        plt.savefig(f'stat/histogram_{TASK_TYPE_MAP[type_task]}_movement.png')
        plt.clf()

histogram()

