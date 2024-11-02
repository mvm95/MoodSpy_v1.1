import pandas as pd

# Elenco delle colonne desiderate
columns = ['SUBJECT', 'MEAN GONOGO TIME', 'MEAN STROOP TIME', 'MEAN T1', 'MEAN T2', 'MEAN T1+T2', 
           'MEAN T3', 'MEAN UNCERTAINITY', 'NUMBER OF STROOP ERROR', 'NUMBER OF GONOGO ERROR', 
           'NUMBER OF UNCERTAINITY']
info_file = pd.read_excel('stat\Status Reclutamento.xlsx', usecols='B,D,F', header=None)
info_file = info_file.iloc[11:, :]
info_file.columns = ['USER', 'TYPE', 'LABELED']
info_file = info_file.loc[info_file['LABELED'] == 'Y', ['USER', 'TYPE']]

movement_list = info_file[info_file['TYPE'] == 'Movement']['USER'].to_list()
movement_list = [f'IDU{int(mov):03d}' for mov in movement_list]

stop_list = info_file[info_file['TYPE'] == 'Stop']['USER'].to_list()
stop_list = [f'IDU{int(stop):03d}' for stop in stop_list]

# Funzione per calcolare le metriche richieste per ogni soggetto
def process_subject_data(df):
    result = []
    
    # Raggruppa per soggetto e itera su ogni soggetto
    grouped = df.groupby('SUBJECT')
    
    for subject, group in grouped:
        subject_data = {'SUBJECT': subject}
        
        # Media per TYPE_OF_TASK specifici
        subject_data['MEAN GONOGO TIME'] = group[group['TYPE_OF_TASK'] == 0]['REACTION_TIME'].mean()
        subject_data['MEAN STROOP TIME'] = group[group['TYPE_OF_TASK'] == 1]['REACTION_TIME'].mean()
        subject_data['MEAN T1'] = group[group['TYPE_OF_TASK'] == 2]['REACTION_TIME'].mean()
        subject_data['MEAN T2'] = group[group['TYPE_OF_TASK'] == 4]['REACTION_TIME'].mean()
        subject_data['MEAN T1+T2'] = group[(group['TYPE_OF_TASK'] == 5) | (group['TYPE_OF_TASK'] == 3)]['REACTION_TIME'].mean()
        subject_data['MEAN T3'] = group[group['TYPE_OF_TASK'] == 3]['REACTION_TIME'].mean()
        
        # Errori per TYPE_OF_TASK 0 e 1
        subject_data['NUMBER OF GONOGO ERROR'] = group[(group['TYPE_OF_TASK'] == 0) & (group['CORRECT'] == 0)].shape[0]
        subject_data['NUMBER OF STROOP ERROR'] = group[(group['TYPE_OF_TASK'] == 1) & (group['CORRECT'] == 0)].shape[0]
        
        # Numero di TYPE_OF_TASK 6
        if subject in movement_list:
            subject_data['NUMBER OF UNCERTAINITY'] = group[group['TYPE_OF_TASK'] == 6].shape[0]
            subject_data['MEAN UNCERTAINITY'] = group[group['TYPE_OF_TASK'] == 6]['REACTION_TIME'].mean()
        else:
            subject_data['NUMBER OF UNCERTAINITY'] = 'N/A'
            subject_data['MEAN UNCERTAINITY'] ='N/A'
        result.append(subject_data)
    
    return pd.DataFrame(result)

# Carica il file CSV
file_path = 'stat/reaction_times.csv'
data = pd.read_csv(file_path)

# Elabora i dati
processed_data = process_subject_data(data)

# Riordina le colonne in base a quelle specificate
processed_data = processed_data[columns]
processed_data = processed_data.applymap(lambda x: f'{x:.3f}' if isinstance(x, float) else x)

# Salva i dati elaborati in un nuovo file CSV
output_file = 'stat/subject_table.csv'
processed_data.to_csv(output_file, index=False)

print(f'File salvato come {output_file}')
