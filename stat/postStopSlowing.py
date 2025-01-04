import os
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu, ttest_ind


def kruskal_test(df):
    groups = [df[df['Previous Go - Video'] == group]['T1'].values for group in df['Previous Go - Video'].unique()]

# Eseguiamo il test di Kruskal-Wallis
    stat, p_value = kruskal(*groups)

    return stat, p_value

def mann_whitney_test(df):
    # Selezioniamo i dati con Previous Go - Video == 1 e > 1
    media = df['Previous Go - Video'].mean()
    group_1 = df[df['Previous Go - Video'] <= media]['T1']
    group_2 = df[df['Previous Go - Video'] > media]['T1']

    # Eseguiamo il test di Mann-Whitney
    # stat, p_value = mannwhitneyu(group_1, group_2, alternative='two-sided')
    stat, p_value = ttest_ind(group_1, group_2)

    return stat, p_value



def obtainMeanT1TimeForNumberOFNoGoStimuli(df, task='Stop'):
    print('********CONSIDERANDO SOLO GLI STIMOLI IN UN VIDEO:*********')
    # median_stop = dfs_stop['Previous Go - Video'].median()
    median_stop = df['Previous Go - Video'].mean()
    # print(dfs_stop['Previous Go - Video'].median(), dfs_stop['Previous Go TOT'].median())
    mean_stop_under_median = df[df['Previous Go - Video'] <= median_stop]['T1'].mean()
    mean_stop_over_median = df[df['Previous Go - Video'] > median_stop]['T1'].mean()
    count_stop_under = df[df['Previous Go - Video'] <= median_stop]['T1'].count()
    count_stop_over = df[df['Previous Go - Video'] > median_stop]['T1'].count()


    print(f'''Gruppo {task}: 
    Mediana Go prima di stimolo no GO : {median_stop}
    Tempo medio T1 per gruppo di stimoli <= della mediana : {mean_stop_under_median} , count = {count_stop_under}
    Tempo medio T1 per gruppo di stimoli > della mediana : {mean_stop_over_median} , count = {count_stop_over}''')

    # print('\n\n*******CONSIDERANDO SOLO GLI STIMOLI IN TOTALE (PIU VIDEO):*******')
    median_stop = df['Previous Go TOT'].mean()
    # print(dfs_stop['Previous Go - Video'].median(), dfs_stop['Previous Go TOT'].median())
    mean_stop_under_median = df[df['Previous Go TOT'] <= median_stop]['T1'].mean()
    count_stop_under = df[df['Previous Go TOT'] <= median_stop]['T1'].count()
    count_stop_over = df[df['Previous Go TOT'] > median_stop]['T1'].count()
    mean_stop_over_median = df[df['Previous Go TOT'] > median_stop]['T1'].mean()

    # print(f'''Gruppo {task}: 
    # Mediana Go prima di stimolo no GO : {median_stop} 
    # Tempo medio T1 per gruppo di stimoli <= della mediana : {mean_stop_under_median}, count = {count_stop_under}
    # Tempo medio T1 per gruppo di stimoli > della mediana : {mean_stop_over_median}, count = {count_stop_over}''')

    print('\n\n')
    print(f'*****GRUPPO {task.upper()}*****')
    print('Tempi medi T1 per numero di stimoli Go che precedeno stimolo No Go (SENZA INTERRUZIONI)')
    for idx in range(1,df['Previous Go - Video'].max()+1):
        mean_stop = df[df['Previous Go - Video'] == idx]['T1'].mean()
        n_sample = df[df['Previous Go - Video'] == idx]['T1'].count()
        print(f'Numero di stimoli precedenti: {idx}, Tempo medio T1: {mean_stop}, Number of samples: {n_sample}')
    # print('\nTempi medi T1 per numero di stimoli Go che precedeno stimolo No Go (CON INTERRUZIONI)')
    # for idx in range(1,df['Previous Go TOT'].max()+1):
    #     mean_stop = df[df['Previous Go TOT'] == idx]['T1'].mean()
    #     n_sample = df[df['Previous Go TOT'] == idx]['T1'].count()
    #     print(f'Numero di stimoli precedenti: {idx}, Tempo medio T1: {mean_stop}, Number of samples: {n_sample}')


def getTraditionalStopSlowingEffect():
    info_file = pd.read_excel('stat\Status Reclutamento.xlsx', usecols='B,D,F', header=None)
    info_file = info_file.iloc[11:, :]
    info_file.columns = ['USER', 'TYPE', 'LABELED']
    info_file = info_file.loc[info_file['LABELED'] == 'Y', ['USER', 'TYPE']]
    movement_list = info_file[info_file['TYPE'] == 'Movement']['USER'].to_list()
    movement_list = [f'IDU{int(mov):03d}' for mov in movement_list]

    stop_list = info_file[info_file['TYPE'] == 'Stop']['USER'].to_list()
    stop_list = [f'IDU{int(stop):03d}' for stop in stop_list]
    df = pd.read_csv('stat/reaction_times.csv')
    df = df[df['TYPE_OF_TASK'] == 0]
    df_stop = df[df['SUBJECT'].isin(stop_list)]
    df_mov = df[df['SUBJECT'].isin(movement_list)]
    #COME FACCIO A CORRELARE I TEMPI DI INIBIZIONE SE IN UN CASO MISURO GO E NELL'ALTRO NO GO?

# getTraditionalStopSlowingEffect()
def kruskalWallisAnalysis(df, task='Stop'):    
    stat_stop, p_value_stop = kruskal_test(df)
    print(f"Kruskal-Wallis test for {task}:")
    print(f"Statistic: {stat_stop}, P-value: {p_value_stop}")

    stat_stop_mw, p_value_stop_mw = mann_whitney_test(df)
    print(f"Mann-Whitney test for {task}:")
    print(f"Statistic: {stat_stop_mw}, P-value: {p_value_stop_mw}")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    print('NEW')
    path = 'stat/subject_tables_Stop'
    # path = 'stat/subject_tables'
    info_file = pd.read_excel('stat\Status Reclutamento.xlsx', usecols='B,D,F', header=None)
    info_file = info_file.iloc[11:, :]
    info_file.columns = ['USER', 'TYPE', 'LABELED']
    info_file = info_file.loc[info_file['LABELED'] == 'Y', ['USER', 'TYPE']]

    movement_list = info_file[info_file['TYPE'] == 'Movement']['USER'].to_list()
    movement_list = [f'IDU{int(mov):03d}' for mov in movement_list]

    stop_list = info_file[info_file['TYPE'] == 'Stop']['USER'].to_list()
    stop_list = [f'IDU{int(stop):03d}' for stop in stop_list]
    dfs_stop = []
    dfs_movement = []
    file_list = os.listdir(path)
    for file in file_list:
        df = pd.read_csv(os.path.join(path,file))
        if file[:6] in movement_list:
            dfs_movement.append(df)
        else:
            dfs_stop.append(df)

    if not len(dfs_stop) == 0:
        dfs_stop = pd.concat(dfs_stop, ignore_index=True)
        dfs_stop = dfs_stop[dfs_stop['T2'] > 200]
        dfs_stop = dfs_stop[dfs_stop['Previous Go - Video'] > 0]
        dfs_stop = dfs_stop[dfs_stop['Previous Go TOT'] > 0]

        median_t1 = dfs_stop['Previous Go - Video'].median()

        # Adding a new column to define the groups
        group_1 = dfs_stop[dfs_stop['Previous Go - Video'] <= median_t1]['T2']
        group_2 = dfs_stop[dfs_stop['Previous Go - Video'] > median_t1]['T2']

        print(group_1.mean(), group_2.mean())

        # Combine into a single DataFrame for plotting
        boxplot_data = pd.DataFrame({
            'Group': ['<= Median'] * len(group_1) + ['> Median'] * len(group_2),
            'T1': pd.concat([group_1, group_2])
        })

        # Plotting the box plots
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Group', y='T1', data=boxplot_data)
        plt.xlabel('Group')
        plt.ylabel('T1 Values')
        plt.title('Box Plot of T1 for <= Median and > Median Groups')
        plt.show()

    if not len(dfs_movement) == 0:
        dfs_movement = pd.concat(dfs_movement, ignore_index=True)
        dfs_movement = dfs_movement[dfs_movement['T2'] > 200]
        dfs_movement = dfs_movement[dfs_movement['Previous Go TOT'] > 0]
        dfs_movement = dfs_movement[dfs_movement['Previous Go - Video'] > 0]
    obtainMeanT1TimeForNumberOFNoGoStimuli(dfs_stop)
    print('\n\n')
    kruskalWallisAnalysis(dfs_stop)
    print('\n\n')
    print('OLD')
    path = 'stat/subject_tables'
    info_file = pd.read_excel('stat\Status Reclutamento.xlsx', usecols='B,D,F', header=None)
    info_file = info_file.iloc[11:, :]
    info_file.columns = ['USER', 'TYPE', 'LABELED']
    info_file = info_file.loc[info_file['LABELED'] == 'Y', ['USER', 'TYPE']]

    movement_list = info_file[info_file['TYPE'] == 'Movement']['USER'].to_list()
    movement_list = [f'IDU{int(mov):03d}' for mov in movement_list]

    stop_list = info_file[info_file['TYPE'] == 'Stop']['USER'].to_list()
    stop_list = [f'IDU{int(stop):03d}' for stop in stop_list]
    dfs_stop = []
    dfs_movement = []
    file_list = os.listdir(path)
    for file in file_list:
        df = pd.read_csv(os.path.join(path,file))
        if file[:6] in movement_list:
            dfs_movement.append(df)
        else:
            dfs_stop.append(df)
    if not len(dfs_stop) == 0:
        dfs_stop = pd.concat(dfs_stop, ignore_index=True)
        dfs_stop = dfs_stop[dfs_stop['T2'] > 200]
        dfs_stop = dfs_stop[dfs_stop['Previous Go - Video'] > 0]
        dfs_stop = dfs_stop[dfs_stop['Previous Go TOT'] > 0]

    if not len(dfs_movement) == 0:
        dfs_movement = pd.concat(dfs_movement, ignore_index=True)
        dfs_movement = dfs_movement[dfs_movement['T2'] > 200]
        dfs_movement = dfs_movement[dfs_movement['Previous Go TOT'] > 0]
        dfs_movement = dfs_movement[dfs_movement['Previous Go - Video'] > 0]
    obtainMeanT1TimeForNumberOFNoGoStimuli(dfs_stop)
    print('\n\n')
    kruskalWallisAnalysis(dfs_stop)


if __name__== '__main__':
    main()