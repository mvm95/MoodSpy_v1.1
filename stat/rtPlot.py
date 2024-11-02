import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

trialSubject = []
# for idx in range(1,10):
#     if not idx == 5:
#         trialSubject.append(f'IDU00{idx}')
#     else:
#         trialSubject.append('IDU105')

def estimateError():
    labelList = pd.read_excel('Etichettatura.xlsx')
    labelList['Subject'] = labelList['Video'].str[:6]
    labelList = labelList[~labelList['Subject'].isin(trialSubject)]
    timeDifferences = []
    for _, row in labelList.iterrows():
        for idx in range(1,13):
            data_tuple = row[f'column_{idx}']
            if not type(data_tuple) == str:
                continue
            data_tuple = eval(data_tuple)
            label, t0, _ = data_tuple
            if label == 3:
                prev_data_tuple = row[f'column_{idx-1}']
                prev_data_tuple = eval(prev_data_tuple)
                _, _, tfPrev = prev_data_tuple
                timeDifferences.append(t0 - tfPrev)
    timeDifferences = np.asanyarray(timeDifferences)
    return np.median(timeDifferences)
    # print(len(timeDifferences))
    # plt.plot(timeDifferences, label = 'Duration')
    # plt.hlines(np.median(timeDifferences),xmin=0,xmax=len(timeDifferences), label = 'Median Value', color = 'red')
    # plt.title(f'Duration of Orange State of Pepper, mean:{np.mean(timeDifferences):.2f}, median:{np.median(timeDifferences):.2f}, std:{np.std(timeDifferences):.2f}')
    # # plt.savefig
    # plt.legend()
    # plt.show()
    # print(timeDifferences.mean(), timeDifferences.std())



taskDict = {
    0 : 'GoNogo Computer',
    1 : 'Stroop Test',
    2 : 'Robot Interaction'
    # 2 : 'Stop after Red Light (Red)',
    # 3 : 'Start after Green Light (Green)'
}

def makePlotReactionTimesOverTime():
    df = pd.read_csv('stat/reaction_times.csv')
    df = df[df['SUBJECT'] != 'IDU000']
    subjects = df['SUBJECT'].unique()
    this_subject = 0
    this_task = 2
    error = estimateError()
    df.loc[(df['SUBJECT'].isin(trialSubject)) & df['REACTION_TIME'] == 3, 'REACTION_TIME'] -= error

    fig,ax = plt.subplots()
    def updatePlt(index_sub, index_task):
        ax.clear()
        sub = subjects[index_sub]
        if index_task == 2:
            suppText = ''
            suppText2 = ''
            if sub in trialSubject:
                suppText = '*'
                suppText2 = '*estimated'
            df2 = df[df['TYPE_OF_TASK'] == 2].drop(columns=['CORRECT'])
            data = df2[df2['SUBJECT'] == sub]
            ax.plot(data['TRIAL'],data['REACTION_TIME'], color = 'red', linewidth = 1)
            mean = np.mean(data['REACTION_TIME'].to_numpy())
            std = np.std(data['REACTION_TIME'].to_numpy())
            ax.hlines(mean, 1, 17, color = 'orange', label = 'NoGo mean +- std')
            ax.hlines(mean - std, 1, 17, color = 'orange', linestyle = '--')
            ax.hlines(mean + std, 1, 17, color = 'orange', linestyle = '--')
            ax.scatter(data['TRIAL'],data['REACTION_TIME'], color = 'red', label = 'stop after NoGo stimulus')
            df2 = df[df['TYPE_OF_TASK'] == 3].drop(columns=['CORRECT'])
            data = df2[df2['SUBJECT'] == sub]
            ax.plot(data['TRIAL'],data['REACTION_TIME'], color = 'green', linewidth = 1)
            mean = np.mean(data['REACTION_TIME'].to_numpy())
            std = np.std(data['REACTION_TIME'].to_numpy())
            ax.hlines(mean, 1, 17, color = 'blue', label = 'Go mean +- std')
            ax.hlines(mean - std, 1, 17, color = 'blue', linestyle = '--')
            ax.hlines(mean + std, 1, 17, color = 'blue', linestyle = '--')
            ax.scatter(data['TRIAL'],data['REACTION_TIME'], color = 'green', label = 'start after Go stimulus' + suppText2) 
            ax.set_title(f'Reaction Times for Subject {sub}, {taskDict[this_task]}'+suppText)
            ax.set_xlabel('Trial')
            ax.set_xticks(range(1, 18))
        else:
            df2 = df[df['TYPE_OF_TASK'] == index_task]
            if int(sub[3:6]) > 100:
                sub = f'IDU00{int(sub[3:6])-100}'
            df2 = df2[df2['SUBJECT'] == sub]
            df2 = df2[~((df2['CORRECT'] == 1) & (df2['REACTION_TIME'] == -1))]
            data = df2[df2['CORRECT'] == 0]
            ax.scatter(data['TRIAL'],data['REACTION_TIME'], color = 'red', label = 'Mistakes')
            data = df2[df2['CORRECT'] == 1]
            ax.scatter(data['TRIAL'],data['REACTION_TIME'], color = 'green', label = 'Right')
            ax.plot(df2['TRIAL'], df2['REACTION_TIME'], color = 'blue', linewidth = 1)
            mean = np.mean(data['REACTION_TIME'].to_numpy())
            std = np.std(data['REACTION_TIME'].to_numpy())
            ax.scatter(data['TRIAL'],data['REACTION_TIME'], color = 'red', label = 'stop after NoGo stimulus')
            ax.legend()
            ax.set_title(f'Reaction Times for Subject {sub}, {taskDict[this_task]}')
            ax.set_xlabel('Trial')
            if this_task == 0:
                ax.hlines(mean, 1, 77, color = 'orange', label = 'Mean +- std')
                ax.hlines(mean - std, 1, 77, color = 'orange', linestyle = '--')
                ax.hlines(mean + std, 1, 77, color = 'orange', linestyle = '--')
                ax.set_xticks(range(1, 78))
            else:
                ax.hlines(mean, 1, 25, color = 'orange', label = 'Mean +- std')
                ax.hlines(mean - std, 1, 25, color = 'orange', linestyle = '--')
                ax.hlines(mean + std, 1, 25, color = 'orange', linestyle = '--')
                ax.set_xticks(range(1,25))
        ax.grid()
        ax.legend()
        ax.set_ylabel('Reaction Time')
        fig.canvas.draw()
    def onKey(event):
        nonlocal this_subject
        nonlocal this_task
        if event.key == 'right':
            this_subject = (this_subject + 1) % len(subjects)
        elif event.key == 'left':
            this_subject = (this_subject - 1) % len(subjects)
        elif event.key == 'up':
            this_task = (this_task + 1) % 3
        elif event.key == 'down':
            this_task = (this_task - 1) % 3
        updatePlt(this_subject, this_task)
    fig.canvas.mpl_connect('key_press_event', onKey)
    updatePlt(this_subject, this_task)
    plt.show()


def updatePlot(ax, df, subjects, index_subject, index_task, error):
    ax.clear()
    sub = subjects[index_subject]
    if index_task == 2:
        suppText = ''
        suppText2 = ''
        if sub in trialSubject:
            suppText = '*'
            suppText2 = '*estimated'
        df2 = df[df['TYPE_OF_TASK'] == 2]
        data = df2[df2['SUBJECT'] == sub]
        ax.plot(data['TRIAL'], data['REACTION_TIME'], color='red', linewidth=1)
        ax.scatter(data['TRIAL'], data['REACTION_TIME'], color='red', label='stop after NoGo stimulus')
        df2 = df[df['TYPE_OF_TASK'] == 3]
        data = df2[df2['SUBJECT'] == sub]
        ax.plot(data['TRIAL'], data['REACTION_TIME'], color='green', linewidth=1)
        ax.scatter(data['TRIAL'], data['REACTION_TIME'], color='green', label='start after Go stimulus' + suppText2)
        ax.set_title(f'Reaction Times for Subject {sub}, {taskDict[index_task]}' + suppText)
        ax.set_xlabel('Trial')
        ax.set_xticks(range(1, 18))
    else:
        df2 = df[df['TYPE_OF_TASK'] == index_task]
        if int(sub[3:6]) > 100:
            sub = f'IDU00{int(sub[3:6])-100}'
        df2 = df2[df2['SUBJECT'] == sub]
        df2 = df2[~((df2['CORRECT'] == 1) & (df2['REACTION_TIME'] == -1))]
        data = df2[df2['CORRECT'] == 0]
        ax.scatter(data['TRIAL'], data['REACTION_TIME'], color='red', label='Mistakes')
        data = df2[df2['CORRECT'] == 1]
        ax.scatter(data['TRIAL'], data['REACTION_TIME'], color='green', label='Right')
        ax.plot(df2['TRIAL'], df2['REACTION_TIME'], color='blue', linewidth=1)
        ax.set_title(f'Reaction Times for Subject {sub}, {taskDict[index_task]}')
        ax.set_xlabel('Trial')
        if index_task == 0:
            ax.set_xticks(range(1, 78))
        else:
            ax.set_xticks(range(1, 25))
    ax.grid()
    # ax.tight_layout()
    ax.set_ylabel('Reaction Time [ms]')
    ax.legend()

def animate(index_subject, index_task, i):
    # index_subject = (i // 10) % len(subjects)
    # index_task = i % 3
    updatePlot(ax, df, subjects, index_subject, index_task, error)
    filename = f"frame_{i:03d}.png"
    plt.tight_layout()
    plt.savefig(filename)  # Save each frame
    return filename,

def makeVideo():
    df = pd.read_csv('stat/reaction_times.csv')
    df = df[df['SUBJECT'] != 'IDU000']
    subjects = df['SUBJECT'].unique()
    error = estimateError()
    df.loc[(df['SUBJECT'].isin(trialSubject)) & (df['REACTION_TIME'] == 3), 'REACTION_TIME'] -= error

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(16, 8))

    # Create animation
    ani = animation.FuncAnimation(fig, animate, frames=60, interval=1000, blit=False)

    # Save the animation
    frames = len(subjects)*3
    i = 0
    for sub in range(len(subjects)):
        for task in range(3):
            animate(sub, task, i)
            i += 1
    from moviepy.editor import ImageSequenceClip

    # Create video from saved frames
    clip = ImageSequenceClip([f"frame_{i:03d}.png" for i in range(frames)], fps=1)
    clip.write_videofile('reaction_times.mp4', codec='libx264', fps=1)

    # Optionally clean up the saved frames
    import os
    for i in range(frames):
        os.remove(f"frame_{i:03d}.png")

if __name__ == '__main__':
    # Read data
    # estimateError()
    makePlotReactionTimesOverTime()
    # makeVideo()

