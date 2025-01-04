import os
import pandas as pd

DATA_PATH = ['d:/Marco', 'E:/Marco']
LABEL_PATH = 'Etichettatura_v2.xlsx'
COLUMNS = ['SUBJECT', 'TRIAL', 'REACTION_TIME', 'CORRECT', 'TYPE_OF_TASK']

if not type(DATA_PATH) == list:
    DATA_PATH = [DATA_PATH]

def getComputerResults(user_list):
    df = []
    for path in DATA_PATH:
        for sub in os.listdir(path):
            if sub in user_list:
                csv_file = os.path.join(path, sub, 'GoNogo_results.csv')
                csv_df = pd.read_csv(csv_file)
                correct_list = csv_df['CORRECT'].to_list()
                reaction_times_list = csv_df['REACTION TIME'].to_list()
                trial_list = csv_df['TRIAL'].to_list()
                for correct, rt, trial in zip(correct_list, reaction_times_list, trial_list):
                    df.append({
                        'SUBJECT' : sub,
                        'TRIAL':trial,
                        'REACTION_TIME': rt,
                        'CORRECT' : correct,
                        'TYPE_OF_TASK' : 0                
                    })
    return pd.DataFrame(df, columns=COLUMNS)

def getStroopResults(user_list):
    df = []
    for path in DATA_PATH:
        for sub in os.listdir(path):
            if sub in user_list:
                csv_file = os.path.join(path, sub, 'Stroop_results.csv')
                csv_df = pd.read_csv(csv_file)
                correct_list = csv_df['CORRECT'].to_list()
                reaction_times_list = csv_df['REACTION TIME'].to_list()
                trial_list = csv_df['TRIAL'].to_list()
                for correct, rt, trial in zip(correct_list, reaction_times_list, trial_list):
                    df.append({
                        'SUBJECT' : sub,
                        'TRIAL':trial,
                        'REACTION_TIME': rt,
                        'CORRECT' : correct,
                        'TYPE_OF_TASK' : 1                
                    })
    return pd.DataFrame(df, columns=COLUMNS)
   
def check_red_counter(count, sub):
    while True:
        if sub == 'IDU010' and count in [2, 3, 12]:
            count += 1
        elif sub == 'IDU012' and count == 3:
            count += 1
        elif sub == 'IDU013' and count == 8:
            count += 1
        elif sub == 'IDU019' and count in [6, 7, 11]:
            count += 1
        elif sub == 'IDU027' and count == 13:
            count += 1
        else:
            break  # Esci se `count` non è più nella lista di controllo
    return count

def getLabelResults(user_list):
    label_df = pd.read_excel(LABEL_PATH)
    df = []
    seen_triples = set()  # Per evitare duplicati

    for sub in user_list:
        sub_label = label_df[label_df['Video'].str.contains(sub)]
        sub_label = sub_label.sort_values(by='Video')
        trial_count_red = 0

        for _, row in sub_label.iterrows():
            for col in range(1, 21):
                data_tuple = row[f'column_{col}']
                if isinstance(data_tuple, str):  # Controlla se è una stringa
                    data_tuple = eval(data_tuple)
                    label, t0, tf = data_tuple
                    rt = tf - t0
                    if label == 1:
                        trial_count_red += 1
                        trial_count_red = check_red_counter(trial_count_red, sub)
                        triple = (sub, trial_count_red, 2)
                        if triple not in seen_triples:
                            seen_triples.add(triple)
                            df.append({
                                'SUBJECT': sub,
                                'TRIAL': trial_count_red,
                                'REACTION_TIME': rt,
                                'CORRECT': 0,
                                'TYPE_OF_TASK': 2
                            })
                    elif label == 2:
                        triple = (sub, trial_count_red, 3)
                        if triple not in seen_triples:
                            seen_triples.add(triple)
                            df.append({
                                'SUBJECT': sub,
                                'TRIAL': trial_count_red,
                                'REACTION_TIME': rt,
                                'CORRECT': 0,
                                'TYPE_OF_TASK': 4
                            })
                    elif label == 4:
                        triple = (sub, trial_count_red, 4)
                        if triple not in seen_triples:
                            seen_triples.add(triple)
                            df.append({
                                'SUBJECT': sub,
                                'TRIAL': trial_count_red,
                                'REACTION_TIME': rt,
                                'CORRECT': 0,
                                'TYPE_OF_TASK': 3
                            })

    return pd.DataFrame(df, columns=COLUMNS)
                    
def create_csv_previous_stimuli(user_list, task):
    reaction_times = pd.read_csv('stat/reaction_times_stop.csv')
    task = 'Stop'
    for hard_disk in DATA_PATH:
        for sub in os.listdir(hard_disk):
            if not sub in user_list:
                continue
            path = os.path.join(hard_disk, sub)
            connection_log = os.path.join(path, task, 'Log_files', 'Connection_log.txt')
            pepper_file = os.path.join(path, task, 'Log_files', f'{sub}_pepper.txt')
            with open(connection_log, 'r') as f:
                lines = f.readlines()
            connection_log_list = []
            checked_video = []
            idx = 0
            for j, line in enumerate(lines):
                if line.startswith('Started'):
                    video = line.split(',')[0].split(' ')[-1].strip()
                    ts_start = int(line.split(',')[1].strip())
                    ts_end = int(lines[j+1].split(',')[1].strip())
                    if video == 'Baseline':
                        continue
                    if video in checked_video:
                        idx = checked_video.index(video)
                        connection_log_list[idx] = (video, ts_start, ts_end)  # Aggiorna i dati
                    else:
                        checked_video.append(video)
                        connection_log_list.append((video, ts_start, ts_end))  # Aggiungi nuovi dati
                        idx += 1

            with open(pepper_file, 'r') as f:
                lines = f.readlines()
            planning_vector = []
            for line in lines:
                if line.startswith('Go') or line.startswith(task) or line.startswith('Pepper'):
                    if not line.startswith('Pepper'):
                        times = line.split(':')[1].strip()
                        t1 = int(times.split('-')[0])
                        t2 = int(times.split('-')[1].split(',')[0])
                        check = False
                        for triple in connection_log_list:
                            if (triple[1] <= t1 <= triple[2])  and (triple[0] in checked_video):
                                check = True
                                break
                        if not check:
                            print(line, t1, triple)
                    else:
                        check = True
                    if check:
                        if line.startswith('Go'):
                            planning_vector.append(0)
                        elif line.startswith(task):
                            planning_vector.append(1)
                        else:
                            planning_vector.append('blue')
            df_rt = reaction_times[(reaction_times['SUBJECT'] == sub) & ((reaction_times['TYPE_OF_TASK'] == 2) | (reaction_times['TYPE_OF_TASK'] == 3) | (reaction_times['TYPE_OF_TASK'] == 4))]
            subject_table_path = os.path.join('stat',f'subject_tables_{task}')
            if not os.path.exists(subject_table_path):
                os.makedirs(subject_table_path)
            n_previous_tot = 0
            n_previous = 0
            previous_list = []
            trial = 0
            for item in planning_vector:
                if item == 'blue':
                    n_previous = 0
                if item == 1:
                    trial += 1
                    previous_list.append((trial, n_previous, n_previous_tot))
                    n_previous_tot = 0
                    n_previous = 0
                if item == 0:
                    n_previous_tot +=1
                    n_previous += 1
            df = []
            for item in previous_list:
                time_2 = 'N/A'
                trial, n_previous, n_previous_tot = item
                time_1_df = df_rt[(df_rt['TRIAL'] == trial) & (df_rt['TYPE_OF_TASK'] == 2)]
                time_1 = time_1_df['REACTION_TIME'].iloc[0] if not time_1_df.empty else 'N/A'
                
                time_2_df = df_rt[(df_rt['TRIAL'] == trial) & (df_rt['TYPE_OF_TASK'] == 4)]
                time_2 = time_2_df['REACTION_TIME'].iloc[0] if not time_2_df.empty else 'N/A'
                # Filter for time_2 and set to 'N/A' if empty
                time_3_df = df_rt[(df_rt['TRIAL'] == trial) & (df_rt['TYPE_OF_TASK'] == 3)]
                time_3 = time_3_df['REACTION_TIME'].iloc[0] if not time_3_df.empty else 'N/A'

                df.append({
                    'SUBJECT': sub,
                    'TRIAL' : trial,
                    'T1' : time_1,
                    'T2' : time_2,
                    'T3' : time_3,
                    'Previous Go - Video' : n_previous,
                    'Previous Go TOT' : n_previous_tot
                })
            df = pd.DataFrame(df)
            df.to_csv(os.path.join(subject_table_path, f'{sub}.csv'), index = False)
def main():
    info_file = pd.read_excel('stat\Status Reclutamento.xlsx', usecols = 'B,D,G', header = None)
    info_file = info_file.iloc[11:,:]
    info_file.columns = ['USER', 'TYPE', 'LABELED']
    # sub_list = info_file['USER'].to_list()
    stop_list = [f'IDU{sub:03d}' for sub in info_file[info_file['TYPE'] == 'Stop']['USER'].to_list()]
    movement_list = [f'IDU{sub:03d}' for sub in info_file[info_file['TYPE'] == 'Movement']['USER'].to_list()]
    df_gonogo = getComputerResults(stop_list)
    df_stroop = getStroopResults(stop_list)
    df_video = getLabelResults(stop_list)
    df = pd.concat([df_gonogo, df_stroop, df_video], ignore_index=True)
    df.to_csv('stat/reaction_times_stop.csv', index = False)

def get_t1plust2():
    path = 'stat\subject_tables_Stop'
    dfs = [df for df in os.listdir(path)]
    for subject in dfs:
        df = pd.read_csv(os.path.join(path,subject))
        df['T1+T2'] = df['T1'] + df['T2']
        print(df['T1'], df['T2'])
        print(df['T1+T2'])
        df = df[['SUBJECT', 'TRIAL', 'T1', 'T2', 'T1+T2', 'T3', 'Previous Go - Video', 'Previous Go TOT']]
        # df.to_csv(os.path.join(path,subject))

def make_goNogo_computer_csv():  
    from subject_performance import calc_iqr, skew, kurtosis
    import numpy as np
    df = pd.read_csv('stat/reaction_times_stop.csv')
    df = df[df['TYPE_OF_TASK'] == 0]
    # df = df[df['REACTION_TIME'] > 150]
    grouped = df.groupby('SUBJECT')
    results = []
    for subject, group in grouped:
        subject_stats = {'SUBJECT' : subject}
        group = group[group['CORRECT'] == 1]
        group = group[group['REACTION_TIME'] > 150]
        times = group['REACTION_TIME'].dropna()
        
        if len(times) > 0 :
            subject_stats[f'RT_mean'] = times.mean()
            subject_stats[f'RT_median'] = times.median()
            subject_stats[f'RT_std'] = times.std()
            subject_stats[f'RT_cv'] = times.std()/ times.mean()
            subject_stats[f'RT_skew'] = skew(times)
            subject_stats[f'RT_kurt'] = kurtosis(times)
            iqr, outlier = calc_iqr(times)
            subject_stats[f'RT_iqr'] = iqr
            subject_stats[f'RT_outlier'] = outlier
            subject_stats['RT_noerrors'] = len(df[(df['CORRECT'] == 0) & (df['SUBJECT'] == subject)])
            subject_stats[f'RT_nsamples'] = len(times)
        else:
            subject_stats[f'RT_mean'] = np.nan
            subject_stats[f'RT_median'] = np.nan
            subject_stats[f'RT_std'] = np.nan
            subject_stats[f'RT_cv'] = np.nan
            subject_stats[f'RT_skew'] = np.nan
            subject_stats[f'RT_kurt'] = np.nan
            iqr, outlier = calc_iqr(times)
            subject_stats[f'RT_iqr'] = np.nan
            subject_stats[f'RT_outlier'] = np.nan
            subject_stats['RT_noerrors'] = np.nan
            subject_stats[f'RT_nsamples'] = 0
        results.append(subject_stats)
    subject_stats = {'SUBJECT' : 'ALL'}
    times = df['REACTION_TIME'].dropna()
    times = times[times > 150]
    if len(times) > 0 :
        subject_stats[f'RT_mean'] = times.mean()
        subject_stats[f'RT_median'] = times.median()
        subject_stats[f'RT_std'] = times.std()
        subject_stats[f'RT_cv'] = times.std()/ times.mean()
        subject_stats[f'RT_skew'] = skew(times)
        subject_stats[f'RT_kurt'] = kurtosis(times)
        iqr, outlier = calc_iqr(times)
        subject_stats[f'RT_iqr'] = iqr
        subject_stats[f'RT_outlier'] = outlier
        subject_stats['RT_noerrors'] = len(df[df['CORRECT'] == 0])
        subject_stats[f'RT_nsamples'] = len(times)
    else:
        subject_stats[f'RT_mean'] = np.nan
        subject_stats[f'RT_median'] = np.nan
        subject_stats[f'RT_std'] = np.nan
        subject_stats[f'RT_cv'] = np.nan
        subject_stats[f'RT_skew'] = np.nan
        subject_stats[f'RT_kurt'] = np.nan
        iqr, outlier = calc_iqr(times)
        subject_stats[f'RT_iqr'] = np.nan
        subject_stats[f'RT_outlier'] = np.nan
        subject_stats['RT_noerrors'] = np.nan
        subject_stats[f'RT_nsamples'] = 0
    results.append(subject_stats)
    results = pd.DataFrame(results)
    results.to_csv('stat/goNogo_computer.csv', index=False)

def make_stroop_times_csv():
    stroop_path = r'C:\Users\marco\OneDrive - unifi.it\Cartella Ricerca Progetto Destini Condisa\Sperimentazione IMAGINE\StroopTests'
    sub_file = [sub for sub in os.listdir('stat/subject_tables_stop')]
    stroop_files = [file for file in os.listdir(stroop_path) if file in sub_file]
    congruent = []
    incongruent = []
    all_stroop = []
    for file in stroop_files:
        sub = file.split('.')[0]
        df_stroop = pd.read_csv(os.path.join(stroop_path, file))
        columns = df_stroop.columns
        df_stroop['SUBJECT'] = sub
        columns = columns.insert(0, 'SUBJECT')
        df_stroop = df_stroop[columns]
        df_congruent = df_stroop[df_stroop['CONGRUENT'] == 1]
        df_incongruent =  df_stroop[df_stroop['CONGRUENT'] == 0]
        congruent.append(df_congruent)
        incongruent.append(df_incongruent)
        all_stroop.append(df_stroop)
    congruent = pd.concat(congruent, ignore_index=True)
    congruent.to_csv('stat/reaction_times_stroop_congruent.csv', index=False)
    incongruent = pd.concat(incongruent, ignore_index=True)
    incongruent.to_csv('stat/reaction_times_stroop_incongruent.csv', index=False)
    all_stroop = pd.concat(all_stroop, ignore_index=True)
    all_stroop.to_csv('stat/reaction_times_stroop_all.csv', index=False)

def make_stroop_csv():
    from subject_performance import calc_iqr, skew, kurtosis
    import numpy as np
    for file in ['congruent', 'incongruent', 'all']:
        file = f'stroop_{file}'
        df = pd.read_csv(f'stat/reaction_times_{file}.csv')
        # df = df[df['TYPE_OF_TASK'] == 0]
        # df = df[df['REACTION_TIME'] > 150]
        grouped = df.groupby('SUBJECT')
        results = []
        for subject, group in grouped:
            subject_stats = {'SUBJECT' : subject}
            group = group[group['CORRECT'] == 1]
            group = group[group['REACTION TIME'] > 150]
            times = group['REACTION TIME'].dropna()
            
            if len(times) > 0 :
                subject_stats[f'RT_mean'] = times.mean()
                subject_stats[f'RT_median'] = times.median()
                subject_stats[f'RT_std'] = times.std()
                subject_stats[f'RT_cv'] = times.std()/ times.mean()
                subject_stats[f'RT_skew'] = skew(times)
                subject_stats[f'RT_kurt'] = kurtosis(times)
                iqr, outlier = calc_iqr(times)
                subject_stats[f'RT_iqr'] = iqr
                subject_stats[f'RT_outlier'] = outlier
                subject_stats['RT_noerrors'] = len(df[(df['CORRECT'] == 0) & (df['SUBJECT'] == subject)])
                subject_stats[f'RT_nsamples'] = len(times)
            else:
                subject_stats[f'RT_mean'] = np.nan
                subject_stats[f'RT_median'] = np.nan
                subject_stats[f'RT_std'] = np.nan
                subject_stats[f'RT_cv'] = np.nan
                subject_stats[f'RT_skew'] = np.nan
                subject_stats[f'RT_kurt'] = np.nan
                iqr, outlier = calc_iqr(times)
                subject_stats[f'RT_iqr'] = np.nan
                subject_stats[f'RT_outlier'] = np.nan
                subject_stats['RT_noerrors'] = np.nan
                subject_stats[f'RT_nsamples'] = 0
            results.append(subject_stats)
        subject_stats = {'SUBJECT' : 'ALL'}
        times = df['REACTION TIME'].dropna()
        times = times[times > 150]
        if len(times) > 0 :
            subject_stats[f'RT_mean'] = times.mean()
            subject_stats[f'RT_median'] = times.median()
            subject_stats[f'RT_std'] = times.std()
            subject_stats[f'RT_cv'] = times.std()/ times.mean()
            subject_stats[f'RT_skew'] = skew(times)
            subject_stats[f'RT_kurt'] = kurtosis(times)
            iqr, outlier = calc_iqr(times)
            subject_stats[f'RT_iqr'] = iqr
            subject_stats[f'RT_outlier'] = outlier
            subject_stats['RT_noerrors'] = len(df[df['CORRECT'] == 0])
            subject_stats[f'RT_nsamples'] = len(times)
        else:
            subject_stats[f'RT_mean'] = np.nan
            subject_stats[f'RT_median'] = np.nan
            subject_stats[f'RT_std'] = np.nan
            subject_stats[f'RT_cv'] = np.nan
            subject_stats[f'RT_skew'] = np.nan
            subject_stats[f'RT_kurt'] = np.nan
            iqr, outlier = calc_iqr(times)
            subject_stats[f'RT_iqr'] = np.nan
            subject_stats[f'RT_outlier'] = np.nan
            subject_stats['RT_noerrors'] = np.nan
            subject_stats[f'RT_nsamples'] = 0
        results.append(subject_stats)
        results = pd.DataFrame(results)
        results.to_csv(f'stat/{file}.csv', index=False)

if __name__ == '__main__': 
    make_stroop_times_csv()
    make_stroop_csv()
    # info_file = pd.read_excel('stat\Status Reclutamento.xlsx', usecols = 'B,D,G', header = None)
    # info_file = info_file.iloc[11:,:]
    # info_file.columns = ['USER', 'TYPE', 'LABELED']
    # # sub_list = info_file['USER'].to_list()
    # stop_list = [f'IDU{sub:03d}' for sub in info_file[info_file['TYPE'] == 'Stop']['USER'].to_list()]
    # create_csv_previous_stimuli(stop_list, task = 'Stop')

