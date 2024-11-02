import os
import pandas as pd

DATA_PATH = ['d:/Marco', 'E:/Marco']
LABEL_PATH = 'Etichettatura.xlsx'

COLUMNS = ['SUBJECT', 'TRIAL', 'REACTION_TIME', 'CORRECT', 'TYPE_OF_TASK']

if not type(DATA_PATH) == list:
    DATA_PATH = [DATA_PATH]

def getComputerGameResult():
    df = []
    for path in DATA_PATH:
        for sub in os.listdir(path):
            sub_file = f'IDU00{int(sub[3:6])-100}' if int(sub[3:6]) > 100 else sub
            csv_file = os.path.join(path, sub_file, 'GoNogo_results.csv')
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
    df = pd.DataFrame(df, columns=COLUMNS)
    return df

def getStroopResult():
    df = []
    for path in DATA_PATH:
        for sub in os.listdir(path):
            sub_file = f'IDU00{int(sub[3:6])-100}' if int(sub[3:6]) > 100 else sub   
            csv_file = os.path.join(path, sub_file, 'Stroop_results.csv')
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
    df = pd.DataFrame(df, columns=COLUMNS)
    return df

def getLabelResult():
    df = []
    label_df = pd.read_excel(LABEL_PATH)
    this_sub = ''
    for _, row in label_df.iterrows():
        sub = row['Video'][:6]
        if not sub == this_sub:
            trial_count_red = 0
            trial_count_green = 0
            this_sub = sub
        print(sub, this_sub)
        detectedTask = False
        for idx in range(1, 13):
            data_tuple = row[f'column_{idx}']
            if type(data_tuple) == str:
                data_tuple = eval(data_tuple)
                label, t0, tf = data_tuple
                if label == 1 or label == 3:
                    rt = tf - t0
                    if label == 1:
                        detectedTask = True
                        trial_count_red += 1
                        df.append({
                            'SUBJECT' : sub,
                            'TRIAL':trial_count_red,
                            'REACTION_TIME': rt,
                            'CORRECT' : -1,
                            'TYPE_OF_TASK' : 2
                        })
                        data_tuple_next = row[f'column_{idx+1}']
                        # print(data_tuple_next)
                        if type(data_tuple_next) == str:
                            data_tuple_next = eval(data_tuple_next)
                            label, t0, tf = data_tuple_next
                            if label == 2:
                                rt_2 = tf - t0
                                df.append({
                                    'SUBJECT' : sub,
                                    'TRIAL':trial_count_red,
                                    'REACTION_TIME': rt_2,
                                    'CORRECT' : -1,
                                    'TYPE_OF_TASK' : 4
                                }) 
                                df.append({
                                    'SUBJECT' : sub,
                                    'TRIAL':trial_count_red,
                                    'REACTION_TIME': rt + rt_2,
                                    'CORRECT' : -1,
                                    'TYPE_OF_TASK' : 5
                                }) 
                    if label == 3:
                        if not detectedTask:
                            trial_count_red += 1
                        trial_count_green += 1
                        df.append({
                            'SUBJECT' : sub,
                            'TRIAL':trial_count_red,
                            'REACTION_TIME': rt,
                            'CORRECT' : -1,
                            'TYPE_OF_TASK' : 3
                        })
                # if (label == 2 and not detectedTask) and (not detectedTask or label == 0)  :
                    # trial_count_red += 1
                    # detectedTask = False
                   
    df = pd.DataFrame(df, columns=COLUMNS)
    return df



def save_csv_file():
    df = pd.concat([getComputerGameResult(), getStroopResult(), getLabelResult()])
    df.to_csv('stat/reaction_times.csv', index=False)


def create_sequence_csv():
    info_file = pd.read_excel('stat\Status Reclutamento.xlsx',usecols = 'B,D,F',header=None)
    info_file = info_file.iloc[11:,:]
    info_file.columns = ['USER','TYPE', 'LABELED']
    info_file = info_file.loc[info_file['LABELED'] == 'Y', ['USER', 'TYPE']]
    sub_list = info_file['USER'].to_list()
    sub_list = [f'IDU{int(sub):03d}' for sub in sub_list]
    type_list = info_file['TYPE'].to_list()
    df = []
    label_df_tot = pd.read_excel(LABEL_PATH)
    video_list = label_df_tot['Video'].to_list()
    tuple_list = [(sub, typ) for sub, typ in zip(sub_list, type_list)]
    tuple_list = sorted(tuple_list, key= lambda x: x[0])
    movement_list = info_file[info_file['TYPE'] == 'Movement']['USER'].to_list()
    movement_list = [f'IDU{int(mov):03d}' for mov in movement_list]

    stop_list = info_file[info_file['TYPE'] == 'Stop']['USER'].to_list()
    stop_list = [f'IDU{int(stop):03d}' for stop in stop_list]
    label_file = pd.read_excel('Etichettatura.xlsx')
    label_file = label_file['Video'].to_list()

    reaction_times = pd.read_csv('stat/reaction_times.csv')
    # print(len(movement_list), len(stop_list))
    for sub in (sub_list):
        if sub in movement_list:
            task = 'Movement'
        elif sub in stop_list:
        # else:
            task = 'Stop'
        else:
            continue
        path = os.path.join(DATA_PATH[0], sub)
        if not os.path.exists(path):
            path = os.path.join(DATA_PATH[1], sub)   
        # leggere connection log
        #leggere pepper log
        # vedi inizio e fine video
        #togli duplicati
        # vedi se il video è in etichettatura
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
        df_rt = reaction_times[(reaction_times['SUBJECT'] == sub) & ((reaction_times['TYPE_OF_TASK'] == 2) | (reaction_times['TYPE_OF_TASK'] == 3))]
        subject_table_path = os.path.join('stat','subject_tables')
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
            
            # Filter for time_2 and set to 'N/A' if empty
            time_2_df = df_rt[(df_rt['TRIAL'] == trial) & (df_rt['TYPE_OF_TASK'] == 3)]
            time_2 = time_2_df['REACTION_TIME'].iloc[0] if not time_2_df.empty else 'N/A'

            df.append({
                'SUBJECT': sub,
                'TRIAL' : trial,
                'T1' : time_1,
                'T2' : time_2,
                'Previous Go - Video' : n_previous,
                'Previous Go TOT' : n_previous_tot
            })
        df = pd.DataFrame(df)
        df.to_csv(os.path.join(subject_table_path, f'{sub}.csv'), index = False)
        # with open(os.path.join(path, task, 'Log_files', f'planning_vector.txt'), 'r') as f:
        #     real_planning_vector = f.readlines()
        # real_planning_vector = [int(n) for n in real_planning_vector]
        # real_planning_vector = [0 if n == 0 else 1 for n in real_planning_vector]



def main():
    info_file = pd.read_excel('stat\Status Reclutamento.xlsx',usecols = 'B,D,F',header=None)
    info_file = info_file.iloc[11:,:]
    info_file.columns = ['USER','TYPE', 'LABELED']
    info_file = info_file.loc[info_file['LABELED'] == 'Y', ['USER', 'TYPE']]
    sub_list = info_file['USER'].to_list()
    type_list = info_file['TYPE'].to_list()
    df = []
    label_df_tot = pd.read_excel(LABEL_PATH)
    tuple_list = [(sub, typ) for sub, typ in zip(sub_list, type_list)]
    tuple_list = sorted(tuple_list, key= lambda x: x[0])
    this_sub = ''
    for tup in tuple_list:
        sub, typ = tup
        sub = f'IDU{sub:03d}'
        label_df = label_df_tot[label_df_tot['Video'].str.contains(sub)]
        if not sub == this_sub:
            trial_count_red = 0
            trial_count_green = 0
            this_sub = sub
        path = os.path.join(DATA_PATH[0], sub)
        if not os.path.exists(path):
            path = os.path.join(DATA_PATH[1], sub)
        gonogo_file = os.path.join(path, 'GoNogo_results.csv')
        stroop_file = os.path.join(path, 'Stroop_results.csv')
        gonogo_df = pd.read_csv(gonogo_file)
        correct_list = gonogo_df['CORRECT'].to_list()
        reaction_times_list = gonogo_df['REACTION TIME'].to_list()
        trial_list = gonogo_df['TRIAL'].to_list()
        for correct, rt, trial in zip(correct_list, reaction_times_list, trial_list):
            df.append({
                'SUBJECT' : sub,
                'TRIAL':trial,
                'REACTION_TIME': rt,
                'CORRECT' : correct,
                'TYPE_OF_TASK' : 0
            })
        stroop_df = pd.read_csv(stroop_file)
        correct_list = stroop_df['CORRECT'].to_list()
        reaction_times_list = stroop_df['REACTION TIME'].to_list()
        trial_list = stroop_df['TRIAL'].to_list()
        for correct, rt, trial in zip(correct_list, reaction_times_list, trial_list):
            df.append({
                'SUBJECT' : sub,
                'TRIAL':trial,
                'REACTION_TIME': rt,
                'CORRECT' : correct,
                'TYPE_OF_TASK' : 1
            })
        # print(sub, this_sub)
        for _, row in label_df.iterrows():
            detectedTask = False
            for idx in range(1, 13):
                data_tuple = row[f'column_{idx}']
                if type(data_tuple) == str:
                    data_tuple = eval(data_tuple)
                    label, t0, tf = data_tuple
                    # if label == 1 or label == 3:
                    rt = tf - t0
                    if label == 2 and not detectedTask:
                        detectedTask = True
                    if label == 1:
                        detectedTask = True
                        trial_count_red += 1
                        df.append({
                            'SUBJECT' : sub,
                            'TRIAL':trial_count_red,
                            'REACTION_TIME': rt,
                            'CORRECT' :-1,
                            'TYPE_OF_TASK' : 2
                        })
                        data_tuple_next = row[f'column_{idx+1}']
                        # print(data_tuple_next)
                        if type(data_tuple_next) == str:
                            data_tuple_next = eval(data_tuple_next)
                            label, t0, tf = data_tuple_next
                            if label == 2:
                                rt_2 = tf - t0
                                df.append({
                                    'SUBJECT' : sub,
                                    'TRIAL':trial_count_red,
                                    'REACTION_TIME': rt_2,
                                    'CORRECT' : -1,
                                    'TYPE_OF_TASK' : 4
                                }) 
                                df.append({
                                    'SUBJECT' : sub,
                                    'TRIAL':trial_count_red,
                                    'REACTION_TIME': rt + rt_2,
                                    'CORRECT' : -1,
                                    'TYPE_OF_TASK' : 5
                                }) 
                    if label == 3:
                        if not detectedTask:
                            detectedTask = True
                            trial_count_red += 1
                        trial_count_green += 1
                        df.append({
                            'SUBJECT' : sub,
                            'TRIAL':trial_count_red,
                            'REACTION_TIME': rt,
                            'CORRECT' : -1,
                            'TYPE_OF_TASK' : 3
                        })
                        detectedTask = False
                    if label >= 4:
                        if not detectedTask == True:
                            trial_count_red += 1
                            detectedTask = True
                        prev_tuple = row[f'column_{idx-1}']
                        prev_tuple = eval(prev_tuple)
                        prev_label, _, _ = prev_tuple
                        if prev_label >=4:
                            continue
                        count = 0
                        j = idx+1
                        while True:
                            data_tuple_uncertainity = row[f'column_{j}']
                            if type(data_tuple_uncertainity) == str:
                                data_tuple_uncertainity = eval(data_tuple_uncertainity)
                                label_u, t0_u, tf_u = data_tuple_uncertainity
                                if label_u < 4:
                                    break
                                count +=1
                                j += 1
                            else:
                                break
                        if count > 0:
                            j = j -2
                            data_tuple_uncertainity = row[f'column_{j}']
                            data_tuple_uncertainity = eval(data_tuple_uncertainity)
                            label_u, t0_u, tf_u = data_tuple_uncertainity
                            rt = tf_u - t0
                            df.append({
                                'SUBJECT' : sub,
                                'TRIAL':trial_count_red,
                                'REACTION_TIME': rt,
                                'CORRECT' : 0,
                                'TYPE_OF_TASK' : 6
                            })
                            if rt> 1000:
                                print(row['Video'],sub)
                                print(data_tuple, data_tuple_uncertainity,rt)
    df = pd.DataFrame(df, columns=COLUMNS)
    df.to_csv('stat/reaction_times.csv', index=False)



if __name__ == '__main__':
    # file = pd.read_csv('stat/reaction_times.csv')
    # file = file[file['SUBJECT'] == 'IDU016']
    # file = file[file['TYPE_OF_TASK'] >= 4]
    # print(len(file))
    # print(file)
    # create_sequence_csv()
    # try:
    main()
    # except Exception as e:
    #     print(e)
    create_sequence_csv()