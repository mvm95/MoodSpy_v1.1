import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

def z_norm(series):
    return (series - series.mean()) / series.std()

def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())
# Function to perform the correlation analysis and display results
task_type_map = {0: 'GONOGO_COMPUTER', 1: 'STROOP_TEST', 2: 'T1', 3: 'T3', 4: 'T2', 5: 'T2+T1'}

class Form():
    def __init__(self, master):
        self.master = master
        master.title("Correlation Analysis")



        self.minMax = tk.BooleanVar()
        self.minMaxNorm_checkbox = tk.Checkbutton(master, text= 'MinMax Normalization', variable=self.minMax)
        self.minMaxNorm_checkbox.pack(pady=5)

        self.zNorm = tk.BooleanVar()
        self.zNorm_checkbox = tk.Checkbutton(master, text= 'z-Normalization', variable=self.zNorm)
        self.zNorm_checkbox.pack(pady=5)

        # Correlation method selection
        self.correlation_method_var = tk.StringVar(value='Kendall')
        correlation_method_label = ttk.Label(master, text="Select Correlation Method:")
        correlation_method_label.pack(pady=5)

        self.correlation_method_combobox = ttk.Combobox(master, textvariable=self.correlation_method_var, values=['Pearson', 'Kendall', 'Spearman'])
        self.correlation_method_combobox.pack(pady=5)

        self.task_type_var = tk.StringVar(value='Movement')
        task_type_label = ttk.Label(master, text="Select Task Type:")
        task_type_label.pack(pady=5)
        self.task_type_combobox = ttk.Combobox(master, textvariable=self.task_type_var, values=['Movement', 'Stop', 'Both'])
        self.task_type_combobox.pack(pady=5)
        # Run analysis button
        correlation_all_button = ttk.Button(master, text="Correlation Groups", command=self.correlation_group)
        correlation_all_button.pack(pady=20)

        # self.time_type_var = tk.StringVar(value='T1')
        # time_type_label = ttk.Label(master, text='Select Time to Correlate')
        # time_type_label.pack(pady=5)
        # self.time_type_combobox = ttk.Combobox(master, text='Correlation Times', command = self.correlation_times )
        # self.time_type_combobox.pack(pady=20)

        df = pd.read_csv('stat/reaction_times.csv')
        df['REACTION_TIME'] = df['REACTION_TIME'].fillna(0)
        df['CORRECT'] = df['CORRECT'].fillna(1)
        df = df[df['CORRECT'] != 0]
        df = df[df['REACTION_TIME'] > 100]
        df = df[df['TYPE_OF_TASK'] <= 5]
        self.df = df

        self.info_file = pd.read_excel('stat\Status Reclutamento.xlsx', usecols='B,D,F', header=None)
        self.info_file = self.info_file.iloc[11:, :]
        self.info_file.columns = ['USER', 'TYPE', 'LABELED']
        self.info_file = self.info_file.loc[self.info_file['LABELED'] == 'Y', ['USER', 'TYPE']]
        
        self.movement_list = self.info_file[self.info_file['TYPE'] == 'Movement']['USER'].to_list()
        self.movement_list = [f'IDU{int(mov):03d}' for mov in self.movement_list]

        self.stop_list = self.info_file[self.info_file['TYPE'] == 'Stop']['USER'].to_list()
        self.stop_list = [f'IDU{int(stop):03d}' for stop in self.stop_list]

        self.correlation_method_combobox = pd.read_csv('stat/reaction_times.csv')

        self.subject_var = tk.StringVar()
        subject_label = ttk.Label(master, text="Enter Subject ID:")
        subject_label.pack(pady=5)

        sub_combo_values = [f'IDU{sub:03d}' for sub in self.info_file['USER'].to_list()]
        sub_combo_values = sorted(sub_combo_values, key= lambda x: int(x[3:6]))
        self.subject_combobox = ttk.Combobox(master, textvariable=self.subject_var, values = sub_combo_values)
        self.subject_combobox.pack(pady=5)

        # Type of task selection for distribution
        self.task_type_distribution_var = tk.StringVar()
        task_distribution_label = ttk.Label(master, text="Select Task Type for Distribution:")
        task_distribution_label.pack(pady=5)

        task_distribution_combobox = ttk.Combobox(master, textvariable=self.task_type_distribution_var,
                                                   values=[task_type_map[idx] for idx in range(len(task_type_map)) ])
        task_distribution_combobox.pack(pady=5)

        # Show distribution button
        show_distribution_button = ttk.Button(master, text="Show Distribution", command=self.show_distribution)
        show_distribution_button.pack(pady=20)
        
        show_distribution_button_all = ttk.Button(master, text="Show Distribution All Subjects", command=self.show_distribution_all)
        show_distribution_button_all.pack(pady=20)


    def show_distribution(self):
        subject = self.subject_var.get()
        task_type= self.task_type_distribution_var.get()
        reverse_task_type_map = {v: k for k, v in task_type_map.items()}

        # Mapping task type index to task name
        task_type_index = reverse_task_type_map[task_type]
        print(task_type_index)
        # Filter DataFrame
        # print(self.df['TYPE_OF_TASK'])
        filtered_df = self.df[(self.df['SUBJECT'] == subject) & (self.df['TYPE_OF_TASK'] == task_type_index)]
        # filtered_df = min_max_normalize(filtered_df)
        print(filtered_df)

        if filtered_df.empty:
            messagebox.showinfo("No Data", "No data found for the selected Subject and Task Type.")
            return
        filtered_df = filtered_df['REACTION_TIME']
        if self.zNorm.get() and  self.minMax_.get():
            messagebox.showerror("NORM ERROR", "SELECT ONLY ONE NORMALIZATION")
            return   
        if self.zNorm.get():
            filtered_df = z_norm(filtered_df)
        if self.minMax.get():
            filtered_df = min_max_normalize(filtered_df)

        # Plot distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(filtered_df, bins=20, kde=True)
        stopmov = 'Movement' if subject in self.movement_list else 'Stop'
        plt.title(f'Distribution of Reaction Times for {subject} in {task_type} \n Type of task: {stopmov}')
        plt.xlabel('Reaction Time [ms]')
        plt.ylabel('Frequency')
        plt.axvline(filtered_df.median(), color='red', linestyle='dashed', linewidth=1, label='Median Reaction Time')
        plt.legend()
        plt.show()

    def show_distribution_all(self):
        subject = self.subject_var.get()
        task_type= self.task_type_distribution_var.get()
        reverse_task_type_map = {v: k for k, v in task_type_map.items()}

        # Mapping task type index to task name
        task_type_index = reverse_task_type_map[task_type]

        # Filter DataFrame
        # print(self.df['TYPE_OF_TASK'])
        print(task_type_index)

        df_stop = self.df[self.df['SUBJECT'].isin(self.stop_list)]
        df_movement = self.df[self.df['SUBJECT'].isin(self.movement_list)]
        type_task = self.task_type_var.get()
        if type_task == 'Movement':
            df = df_movement 
        elif type_task == 'Stop':
            df = df_stop
        else:
            df = self.df
        print(type_task)
        print(df)
        filtered_df = df[df['TYPE_OF_TASK'] == task_type_index]
        print(df['TYPE_OF_TASK'] )
        print(filtered_df)

        if filtered_df.empty:
            messagebox.showinfo("No Data", "No data found for the selected Subject and Task Type.")
            return
        filtered_df = filtered_df['REACTION_TIME']
        if self.zNorm.get() and  self.minMax.get():
            messagebox.showerror("NORM ERROR", "SELECT ONLY ONE NORMALIZATION")
            return   
        if self.zNorm.get():
            filtered_df = z_norm(filtered_df)
        if self.minMax.get():
            filtered_df = min_max_normalize(filtered_df)
        # Plot distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(filtered_df, bins=20, kde=True)
        # stopmov = 'Movement' if subject in self.movement_list else 'Stop'
        plt.title(f'Distribution of Reaction Times for all subjects in {task_type} \n Type of task: {type_task}')
        plt.xlabel('Reaction Time [ms]')
        plt.ylabel('Frequency')
        plt.axvline(filtered_df.median(), color='red', linestyle='dashed', linewidth=1, label='Median Reaction Time')
        plt.legend()
        plt.show()


    def correlation_group(self):
    # Get user selections
        type_task = self.task_type_var.get()
        correlation_method = self.correlation_method_var.get()
        
        # Load data

        df_stop = self.df[self.df['SUBJECT'].isin(self.stop_list)]
        df_movement = self.df[self.df['SUBJECT'].isin(self.movement_list)]
        if type_task == 'Movement':
            df = df_movement 
        elif type_task == 'Stop':
            df = df_stop
        else:
            df = self.df
        if self.zNorm.get() and  self.minMax_.get():
            messagebox.showerror("NORM ERROR", "SELECT ONLY ONE NORMALIZATION")
            return   
        if self.zNorm.get():
            df['REACTION_TIME'] = z_norm(df['REACTION_TIME'])
        if self.minMax.get():
            df['REACTION_TIME'] = min_max_normalize(df['REACTION_TIME'])

        # Group and pivot the data
        grouped_df = df.groupby(['SUBJECT', 'TYPE_OF_TASK']).median()
        pivoted_df = grouped_df.reset_index().pivot(index='SUBJECT', columns='TYPE_OF_TASK', values='REACTION_TIME')
        pivoted_df = pivoted_df.dropna()
        correlation_matrix = pivoted_df.corr()
        # print('NUMBER OF SUB: ' + str(len(list(set(pivoted_df_filled['SUBJECT'].to_list())))))

        correlation_matrix.columns = ['GONO_COMPUTER', 'STROOP_TEST', 'T1', 'T3', 'T2', 'T2+T1']
        correlation_matrix.index = ['GONO_COMPUTER', 'STROOP_TEST', 'T1', 'T3', 'T2', 'T2+T1']


        # Create a p-value matrix
        p_value_matrix = np.zeros(correlation_matrix.shape)
        correlation_matrix_df = np.zeros_like(correlation_matrix)

        for i in range(correlation_matrix.shape[0]):
            for j in range(correlation_matrix.shape[1]):
                if i != j:
                    # Perform correlation test
                    if correlation_method == 'Pearson':
                        corr, p_value = stats.pearsonr(pivoted_df.iloc[:, i], pivoted_df.iloc[:, j])
                    elif correlation_method == 'Kendall':
                        corr, p_value = stats.kendalltau(pivoted_df.iloc[:, i], pivoted_df.iloc[:, j])
                    else:  # Spearman
                        corr, p_value = stats.spearmanr(pivoted_df.iloc[:, i], pivoted_df.iloc[:, j])

                    correlation_matrix_df[i, j] = corr
                    p_value_matrix[i, j] = p_value
                else:
                    correlation_matrix_df[i, j] = 1
                    p_value_matrix[i, j] = np.nan  # Diagonal elements set to NaN

        # Create DataFrames for correlation and p-values
        correlation_matrix_df = pd.DataFrame(correlation_matrix_df, index=correlation_matrix.index, columns=correlation_matrix.columns)
        p_value_df = pd.DataFrame(p_value_matrix, index=correlation_matrix.index, columns=correlation_matrix.columns)

        # Reorder the matrices
        new_column_order = ['GONO_COMPUTER', 'STROOP_TEST', 'T1', 'T2', 'T2+T1', 'T3']
        correlation_matrix_df = correlation_matrix_df.reindex(index=new_column_order, columns=new_column_order)
        p_value_df = p_value_df.reindex(index=new_column_order, columns=new_column_order)

        # Prepare results for display
        results_df = pd.DataFrame(index=correlation_matrix_df.index)
        for col in correlation_matrix_df.columns:
            results_df[col] = correlation_matrix_df[col].apply(lambda x: f"{x:.3f}") + \
                            ' (p=' + p_value_df[col].apply(lambda x: f"{x:.3f}" if not np.isnan(x) else "NaN") + ')'

        # Plotting the results table
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.axis('tight')
        ax.axis('off')

        table_data = results_df.values
        columns = new_column_order
        table = ax.table(cellText=table_data, colLabels=columns, rowLabels=columns, loc='center', cellLoc='center')

        # Set table properties
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        # Highlight p-values < 0.05 in bold red
        for i in range(len(columns)):
            for j in range(len(columns)):
                if p_value_df.iloc[i, j] < 0.05:
                    table[i + 1, j ].set_text_props(weight='bold', color='red')
                if i<j:
                    table[i + 1, j ].set_text_props(text='')

        plt.title(f'Correlation Coefficients with P-values \n Type of Reaction Time: {type_task}, Method: {correlation_method}', fontsize=14)
        plt.tight_layout()
        plt.show()

# Create main application window
if __name__ == '__main__':
    root = tk.Tk()
    app = Form(root)
    root.mainloop()