import time
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu

SUBJECT = ''
TASK_TYPE = 'BOTH'
TASK_TYPE_MAP = {0: 'GONOGO_COMPUTER', 1: 'STROOP_TEST', 2: 'T1', 3: 'T3', 4: 'T2', 5: 'T2+T1'}
TASK_TYPE_LIST = ['GONO']

def z_norm(series):
    return (series - series.mean()) / series.std()

def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())

def t_test_among_times():
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
    df = df[df['CORRECT'] != 0]
    df = df[df['REACTION_TIME'] > 100]
    df_stop = df[df['SUBJECT'].isin(stop_list)]
    df_movement = df[df['SUBJECT'].isin(movement_list)]

    for type_task in ['MOVEMENT', 'STOP', 'BOTH']:
        type_task = 'STOP'
        results = np.zeros((6,6))
        results_p = np.zeros_like(results)
        if type_task == 'MOVEMENT':
            df2 = df_movement
        elif type_task == 'STOP':
            df2 = df_stop
        else:
            df2 = df
        for i in range(6):
            for j in range(i+1,6):
                # df_task_1 = df2[df2['TYPE_OF_TASK'] == i]
                # df_task_2 = df2[df2['TYPE_OF_TASK'] == j]
                reaction_times_1 = df2['REACTION_TIME']
                # reaction_times_2 = df_task_2['REACTION_TIME']
                aaa = df2[df2['SUBJECT'] == 'IDU002']
                aaa = aaa[aaa['TYPE_OF_TASK'] == 2]
                print(aaa)
                print(df2['REACTION_TIME'].min())
                print(df2['REACTION_TIME'].max())
                df2['REACTION_TIME'] = min_max_normalize(df2['REACTION_TIME'])
                aaa = df2[df2['SUBJECT'] == 'IDU002']
                aaa = aaa[aaa['TYPE_OF_TASK'] == 2]
                print(aaa)
                break
            break
        break
        #         # print(reaction_times_1)
        #         # print(reaction_times_2)
        #         reaction_times_1 = min_max_normalize(reaction_times_1)
        #         reaction_times_2 = min_max_normalize(reaction_times_2)
        #         # reaction_times_1 = z_norm(reaction_times_1)
        #         # reaction_times_2 = z_norm(reaction_times_2)
        #         if len(reaction_times_1) > 1 and len(reaction_times_2) > 1:
        #             # t_stat, p_value = ttest_ind(reaction_times_1, reaction_times_2, equal_var=False)
        #             t_stat, p_value = mannwhitneyu(reaction_times_1, reaction_times_2)
        #             results[j,i] = t_stat
        #             results_p[j,i] = p_value
        #             results[i,j] = t_stat
        #             results_p[i,j] = p_value
        #             # print(f"T-test {TASK_TYPE_MAP[i]} vs {TASK_TYPE_MAP[j]} ({type_task}): t-stat = {t_stat:.2f}, p-value = {p_value:.5f}")
        #         else:
        #             print(f"Not enough data for task type comparison: {TASK_TYPE_MAP[i]} vs {TASK_TYPE_MAP[j]}")
        # results_df = pd.DataFrame(results, index =  ['GONO_COMPUTER', 'STROOP_TEST', 'T1', 'T3', 'T2', 'T2+T1'])
        # # print(results)
        # # print(results_df)
        # results_df.columns = results_df.index
        # p_df = pd.DataFrame(results_p, index= results_df.index, columns=results_df.columns)
        # new_column_order = ['GONO_COMPUTER', 'STROOP_TEST', 'T1', 'T2', 'T2+T1', 'T3']
        # results_df = results_df.reindex(index=new_column_order, columns=new_column_order)
        # # print(results_df)
        # p_df = p_df.reindex(index=new_column_order, columns=new_column_order)
        # table_df = pd.DataFrame(index = results_df.index)
        # for col in results_df.columns:
        #         table_df[col] = results_df[col].apply(lambda x: f"{x:.2f}") + \
        #                         ' (p=' + p_df[col].apply(lambda x: f"{x:.2f}" if not np.isnan(x) else "NaN") + ')'
        # fig, ax = plt.subplots(figsize=(10, 7))
        # ax.axis('tight')
        # ax.axis('off')
        # table_data = table_df.values
        # columns = new_column_order
        # table = ax.table(cellText=table_data, colLabels=columns, rowLabels=columns, loc='center', cellLoc='center')

        # # Set table properties
        # table.auto_set_font_size(False)
        # table.set_fontsize(10)
        # table.scale(1.2, 1.2)

        # # Highlight p-values < 0.05 in bold red
        # for i in range(len(columns)):
        #     for j in range(len(columns)):
        #         if p_df.iloc[i, j] < 0.05:
        #             table[i + 1, j ].set_text_props(weight='bold')
        #         if i<j:
        #             table[i + 1, j ].set_text_props(text='')

        # plt.title(f'T-Test distribution {type_task}', fontsize=14)
        # plt.tight_layout()
        # plt.savefig(f'stat/manwhitney_{type_task}_zNorm.png')
        # plt.clf()
        # print(results)
        # print(results_p)

def t_test_among_task_type():
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
    df = df[df['CORRECT'] != 0]
    df = df[df['REACTION_TIME'] > 100]
    df_stop = df[df['SUBJECT'].isin(stop_list)]
    df_movement = df[df['SUBJECT'].isin(movement_list)]

    task_type_dict = {0 : 'MOVEMENT', 1 : 'STOP', 2 : 'BOTH'}

    def select_type_df(type_task):
        if type_task == 'MOVEMENT':
            # print('a')
            return df_movement
        elif type_task == 'STOP':
            # print('b')
            return df_stop
        else:
            # print('c')
            return df

    for task_type in range(6):
        results = np.zeros((3,3))
        results_p = np.zeros_like(results)
        for i in range(3):
            for j in range(i+1,3):
                df1 = select_type_df(task_type_dict[i])
                df2 = select_type_df(task_type_dict[j])
                df1 = df1[df1['TYPE_OF_TASK'] == task_type]
                df2 = df2[df2['TYPE_OF_TASK'] == task_type]
                reaction_times_1 = df1['REACTION_TIME']
                reaction_times_2 = df2['REACTION_TIME']
                reaction_times_1 = min_max_normalize(reaction_times_1)
                reaction_times_2 = min_max_normalize(reaction_times_2)
                # reaction_times_1 = z_norm(reaction_times_1)
                # reaction_times_2 = z_norm(reaction_times_2)

                if len(reaction_times_1) > 1 and len(reaction_times_2) > 1:
                    # t_stat, p_value = ttest_ind(reaction_times_1, reaction_times_2, equal_var=False)
                    t_stat, p_value = mannwhitneyu(reaction_times_1, reaction_times_2)

                    results[j,i] = t_stat
                    results_p[j,i] = p_value
                    results[i,j] = t_stat
                    results_p[i,j] = p_value

                    print(f"T-test {i} vs {j} {TASK_TYPE_MAP[task_type]} ,value = {t_stat}, p-value = {p_value:.5f}")


        results_df = pd.DataFrame(results, index =  ['MOVEMENT', 'STOP', 'BOTH'])
        results_df.columns = results_df.index
        p_df = pd.DataFrame(results_p, index= results_df.index, columns=results_df.columns)
        table_df = pd.DataFrame(index = results_df.index)
        for col in results_df.columns:
                table_df[col] = results_df[col].apply(lambda x: f"{x:.2f}") + \
                                ' (p=' + p_df[col].apply(lambda x: f"{x:.2f}" if not np.isnan(x) else "NaN") + ')'
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.axis('tight')
        ax.axis('off')

        table_data = table_df.values
        table = ax.table(cellText=table_data, colLabels=results_df.columns, rowLabels=results_df.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        for i in range(len(results_df.columns)):
            for j in range(len(results_df.columns)):
                if p_df.iloc[i, j] < 0.05:
                    table[i + 1, j ].set_text_props(weight='bold')
                if i<j:
                    table[i + 1, j ].set_text_props(text='')

        plt.title(f'T-Test distribution for reaction times {TASK_TYPE_MAP[task_type]}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'stat/manwhit_{TASK_TYPE_MAP[task_type]}_zNorm.png')
        plt.clf()


if __name__ == '__main__':
    t_test_among_times()
    # t_test_among_task_type()
    # info_file = pd.read_excel('stat\Status Reclutamento.xlsx', usecols='B,D,F', header=None)
    # info_file = info_file.iloc[11:, :]
    # info_file.columns = ['USER', 'TYPE', 'LABELED']
    # info_file = info_file.loc[info_file['LABELED'] == 'Y', ['USER', 'TYPE']]

    # movement_list = info_file[info_file['TYPE'] == 'Movement']['USER'].to_list()
    # movement_list = [f'IDU{int(mov):03d}' for mov in movement_list]

    # stop_list = info_file[info_file['TYPE'] == 'Stop']['USER'].to_list()
    # stop_list = [f'IDU{int(stop):03d}' for stop in stop_list]

    # df = pd.read_csv('stat/reaction_times.csv')
    # # df['REACTION_TIME'] = df['REACTION_TIME'].fillna(0)
    # df['CORRECT'] = df['CORRECT'].fillna(1)
    # df = df[df['CORRECT'] != 0]
    # df = df[df['REACTION_TIME'] > 100]
    # df_stop = df[df['SUBJECT'].isin(stop_list)]
    # df_movement = df[df['SUBJECT'].isin(movement_list)]

    # task_type_dict = {0 : 'MOVEMENT', 1 : 'STOP', 2 : 'BOTH'}
    # def select_type_df(type_task):
    #     if type_task == 'MOVEMENT':
    #         # print('a')
    #         return df_movement
    #     elif type_task == 'STOP':
    #         # print('b')
    #         return df_stop
    #     else:
    #         # print('c')
    #         return df
    # for task_type in range(6):
    #     # for i in range(3):
    #     #     for j in range(i+1,3):
    #             i = 0
    #             j = 1
    #             df1 = select_type_df(task_type_dict[i])
    #             df2 = select_type_df(task_type_dict[j])
    #             df1 = df1[df1['TYPE_OF_TASK'] == task_type]
    #             df2 = df2[df2['TYPE_OF_TASK'] == task_type]
    #             df1.loc[:,'REACTION_TIME'] = min_max_normalize(df1.loc[:,'REACTION_TIME'].astype(float))
    #             df2.loc[:,'REACTION_TIME'] = min_max_normalize(df2.loc[:,'REACTION_TIME'].astype(float))
    #             df1 = df1.groupby(['SUBJECT']).agg({'REACTION_TIME': 'median'})
    #             df2 = df2.groupby(['SUBJECT']).agg({'REACTION_TIME': 'median'})

    #             # df1 = df1.reset_index().pivot(index = 'SUBJECT', columns='TYPE_OF_TASK', values='REACTION_TIME')
    #             # df2 = df2.reset_index().pivot(index = 'SUBJECT', columns='TYPE_OF_TASK', values='REACTION_TIME')
    #             # df1 = df1.dropna()
    #             # df2 = df2.dropna()
    #             reaction_times_1 = df1['REACTION_TIME'].to_numpy()
    #             reaction_times_2 = df2['REACTION_TIME'].to_numpy()
    #             # reaction_times_1 = min_max_normalize(reaction_times_1)
    #             # reaction_times_2 = min_max_normalize(reaction_times_2)
    #             # reaction_times_1 = z_norm(reaction_times_1)
    #             # reaction_times_2 = z_norm(reaction_times_2)
    #             corr, p_value = stats.kendalltau(reaction_times_1, reaction_times_2)


    #             print(f"Correlation {task_type_dict[i]} vs {task_type_dict[j]} {TASK_TYPE_MAP[task_type]} ,value = {corr}, p-value = {p_value:.5f}")
