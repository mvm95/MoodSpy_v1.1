import os
from matplotlib import font_manager
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, balanced_accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_dataframe_for_pca(save = True,
                          big_5 = True,
                          bis_11 = True,
                          gonogo = True,
                          stroop_c = True,
                          stroop_i = True):
    big5_features = ['Extraversion', 'Neuroticism', 'Openness']
    bis11_features = ['Attenzione', 'Comp_motorio', 'Autocontrollo', 'Compl_cogn', 'Impulsivita_mot', 'Impulsivita_non_pian']
    gonogo_features = ['RT_std', 'RT_skew', 'RT_kurt', 'RT_noerrors']
    stroop_congruent_features = ['RT_mean', 'RT_median', 'RT_cv', 'RT_iqr', 'RT_outlier']
    stroop_incongruent_features = ['RT_mean', 'RT_median', 'RT_std', 'RT_skew', 'RT_iqr', 'RT_outlier']

    subject_list = [sub.split('.')[0] for sub in os.listdir('stat/subject_tables_Stop') if not 'IDU040' in sub]
    df_path = 'stat/times'

    df_tot = pd.DataFrame({'SUBJECT' : subject_list})

    if big_5:
        df_big5 = pd.read_csv(os.path.join(df_path, 'BIG-5_values.csv'))
        df_big5 = df_big5[df_big5['subject'].isin(subject_list)]
        df_big5.rename(columns={'subject':'SUBJECT'}, inplace=True)
        df_big5 = df_big5[['SUBJECT'] + big5_features]
        df_tot = df_tot.merge(df_big5, on='SUBJECT')

    if bis_11:
        df_bis11 = pd.read_csv(os.path.join(df_path, 'BIS-11_values.csv'))
        df_bis11 = df_bis11[df_bis11['subject'].isin(subject_list)]
        df_bis11.rename(columns={'subject':'SUBJECT'}, inplace=True)
        df_bis11 = df_bis11[['SUBJECT'] + bis11_features]  
        df_tot = df_tot.merge(df_bis11, on='SUBJECT')

    if gonogo:
        df_gonogo = pd.read_csv(os.path.join(df_path, 'goNogo_computer.csv'))
        df_gonogo = df_gonogo[df_gonogo['SUBJECT'].isin(subject_list)]
        df_gonogo = df_gonogo[['SUBJECT'] + gonogo_features]
        df_gonogo.rename(columns={col: f"{col}_gonogo" for col in gonogo_features}, inplace=True)
        df_tot = df_tot.merge(df_gonogo, on='SUBJECT')

    if stroop_c:
        df_stroop_cong= pd.read_csv(os.path.join(df_path, 'stroop_congruent.csv'))
        df_stroop_cong = df_stroop_cong[df_stroop_cong['SUBJECT'].isin(subject_list)]
        df_stroop_cong = df_stroop_cong[['SUBJECT'] + stroop_congruent_features]
        df_stroop_cong.rename(columns={col: f"{col}_stroop_cong" for col in stroop_congruent_features}, inplace=True)
        df_tot = df_tot.merge(df_stroop_cong, on='SUBJECT')

    if stroop_i:
        df_stroop_incong= pd.read_csv(os.path.join(df_path, 'stroop_incongruent.csv'))
        df_stroop_incong = df_stroop_incong[df_stroop_incong['SUBJECT'].isin(subject_list)]
        df_stroop_incong = df_stroop_incong[['SUBJECT'] + stroop_incongruent_features]
        df_stroop_incong.rename(columns={col: f"{col}_stroop_incong" for col in stroop_incongruent_features}, inplace=True)
        df_tot = df_tot.merge(df_stroop_incong, on='SUBJECT')

    if save:
        save_path = 'stat/pca'
        if not os.path.exists(save_path): os.makedirs(save_path)
        df_tot.to_csv(os.path.join(save_path, 'pca_df.csv'))
    return df_tot

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


def find_optimal_cluster(pca_result, min_k=2, max_k = 17):

    silhouette_scores = []
    kmeans_list = []
    for k in range(min_k, max_k):
        kmeans = KMeans(n_clusters=k, n_init=50)
        # kmeans =  KMeans(n_clusters=k)
        kmeans.fit(pca_result)
        score = silhouette_score(pca_result, kmeans.labels_)
        kmeans_list.append(kmeans)
        silhouette_scores.append(score)

    # plt.figure(figsize=(8, 6))
    # plt.plot(range(min_k, max_k), silhouette_scores, marker='o')
    # plt.title('Silhouette Scores for Different Number of Clusters')
    # plt.xlabel('Number of Clusters (k)')
    # plt.ylabel('Silhouette Score')
    # savepath = 'stat/pca/silhouette_score.png'
    # plt.savefig(savepath)
    # plt.show()
    
    # Find the optimal number of clusters (the one with the highest Silhouette Score)
    optimal_k = range(min_k, max_k + 1)[np.argmax(silhouette_scores)]
    optimal_k=2
    print(f"Optimal number of clusters: {optimal_k}")
    
    return optimal_k, kmeans_list[optimal_k - 2]

def pca(big_5 = True,
        bis_11 = True,
        gonogo = True,
        stroop_c = True,
        stroop_i = True,
        ignore_some_subjects = True):
    df = get_dataframe_for_pca(True, big_5,bis_11, gonogo, stroop_c, stroop_i)
    if ignore_some_subjects:
        df = df[df['SUBJECT'] != 'IDU037']
    matrix = df.drop('SUBJECT', axis = 1).to_numpy()

    # print(matrix.shape)
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    # print(mean.shape, std.shape)
    # Apply Z-norm (Z-score normalization)
    # matrix = (matrix - mean) / std
    matrix = scaler.fit_transform(matrix)
    subjects = df['SUBJECT']
    print(subjects)
    columns = df.columns[1:]
    pca = PCA()
    pca.fit(matrix)
    pca_result = pca.fit_transform(matrix)
    loadings = pca.components_
    # loadings_percentages = 100 * np.abs(loadings) / np.sum(np.abs(loadings), axis=1, keepdims=True)
    # pca.components_ = loadings_percentages
    n_cluster, kmeans = find_optimal_cluster(pca_result[:,:3], max_k = matrix.shape[0])
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    n_components = np.argmax(cumulative_variance >= 0.95) + 1
    # plt.bar(range(1, n_components+1), explained_variance_ratio, alpha=0.7, color='blue', label='Varianza spiegata')
    # plt.step(range(1, n_components+1), explained_variance_ratio.cumsum(), where='mid', color='red', label='Varianza cumulativa')
    # plt.xlabel('Componenti principali')
    # plt.ylabel('Varianza spiegata')
    # plt.legend(loc='best')
    # plt.title('Varianza spiegata per componente')
    # plt.show()
    # print(explained_variance_ratio)
    print(cumulative_variance, n_components)
    loadings = pca.components_
    top_features_pc1 = sorted(zip(columns, loadings[0]), key = lambda x: abs(x[1]), reverse=True)
    for idx in range(n_components):
        sum = 0
        for feature, value in sorted(zip(columns, loadings[idx]), key = lambda x: abs(x[1]), reverse=True):
            if sum<75:
                sum += value
                # print(sum)
    cluster_labels = kmeans.labels_
    pca_result = pca_result[:, :n_components]
    # cluster_labels[-2] = 2
    fig = plt.figure()
    ax = fig.add_subplot(111, projection= '3d')
    cmap = plt.cm.jet
    unique_labels = np.unique(cluster_labels)
    norm = plt.Normalize(vmin=min(unique_labels), vmax=max(unique_labels))

    # Scatter plot with cluster labels (colors)
    colors = [cmap(norm(label)) for label in cluster_labels]
    ax.scatter(pca_result[:,0], pca_result[:,1], pca_result[:,2], c=colors, marker = 'o')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    
    # Optionally label each point with its subject
    for i, subject in enumerate(subjects):
        ax.text(pca_result[i, 0], pca_result[i, 1], pca_result[i, 2], subject, size=8)
    # unique_labels = np.unique(cluster_labels)
    handles = []
    for idx, label in enumerate(unique_labels):
        color = cmap(norm(label))
        # Create a dummy plot entry for each unique label
        handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,#markerfacecolor=plt.cm.jet(label / len(unique_labels)), 
                            markersize=10, label=f'Cluster {label}')
        handles.append(handle)
    ax.legend(handles=handles, title="Clusters")
    plt.title("PCA: Subjects in 3D")
    plt.draw()
    fig = plt.figure()
    plt.scatter(pca_result[:,0], pca_result[:,1], c=cluster_labels, marker = 'o')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    # Optionally label each point with its subject
    for i, subject in enumerate(subjects):
        plt.text(pca_result[i, 0], pca_result[i, 1], subject, size=8)
    
    plt.title("PCA: Subjects Clusters")


    plt.draw()
    df = {'SUBJECT' : subjects.to_list(),
          'LABEL' : cluster_labels}
    # print(df)
    df = pd.DataFrame(df)
    df.to_csv('stat/pca/label.csv', index=False)
    # plt.show()
    return pca_result, subjects, cluster_labels, pca

def svm():
    from sklearn.svm import SVC
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import accuracy_score, classification_report
    df = pd.read_csv('stat/times/summary_times_stop_total.csv')
    # df = df[df['SUBJECT'] != 'IDU037']
    labels = pd.read_csv('stat/pca/label.csv')
    subjects = labels['SUBJECT']
    labels = labels['LABEL'].to_numpy()
    df = df[df['SUBJECT'].isin(subjects)]
    times = df.drop('SUBJECT', axis=1)
    columns = times.columns
    columns = [t for t in times.columns if not 'T3' in t]
    columns = [t for t in columns if not 'sample' in t]
    # columns = ['T1_std', 'T1_cv', 'T1_skewness', 'T1_kurtosis', 'T1_outlier', 'T2_iqr' ] #BIS11
    # columns.extend(['T1_median', 'T1_std', 'T1_kurtosis', 'T2_cv'])
    # columns = list(set(columns))
    times = times[columns]
    times = scaler.fit_transform(times)
    param_grid = {
        'C': [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': [0.01, 0.02,0.05, 0.1, 0.2, 0.5, 1,2, 5,10,20,50,100]
    }
    # param_grid = {
    #     'C': [0.1, 1,10, 100],
    #     'kernel': ['linear', 'rbf', 'poly'],
    #     'gamma': [0.01, 0.1, 1,10,100]
    # }
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='balanced_accuracy')
    grid_search.fit(times, labels)
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    # Initialize SVM with best parameters
    svm_model = SVC(**best_params)
    # pca = PCA()
    # pca.fit(times)
    # pca_result = pca.fit_transform(times)
    # explained_variance_ratio = pca.explained_variance_ratio_
    # cumulative_variance = np.cumsum(explained_variance_ratio)
    # n_components = np.argmax(cumulative_variance >= 0.95) + 1
    # print(n_components)
    # Inizializzazione del modello e Leave-One-Out
    loo = LeaveOneOut()
    # svm_model = SVC(kernel='linear', C=1.0)  # Puoi cambiare kernel o C se necessario

    y_true = []
    y_pred = []
    # Validazione Leave-One-Out
    for train_index, test_index in loo.split(times):
        X_train, X_test = times[train_index], times[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Addestramento del modello
        svm_model.fit(X_train, y_train)

        # Predizione
        prediction = svm_model.predict(X_test)

        # Salvare i risultati
        y_true.append(y_test[0])  # Essendo LOO, test_index contiene sempre un solo elemento
        y_pred.append(prediction[0])

    # Calcolo delle metriche
    accuracy = accuracy_score(y_true, y_pred)
    print(f'BALANCED ACCURACY: {balanced_accuracy_score(y_true, y_pred)}')
    print(f"Accuracy (LOO): {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print(y_true, y_pred)
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Generate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_true))
    disp.plot(cmap='Reds', values_format='d')  # You can change the colormap if needed
    for c in disp.ax_.collections:
        if isinstance(c, plt.cm.ScalarMappable):  # Check if it's a colorbar object
            c.colorbar.set_visible(False)
    plt.xlabel('Predicted Labels', fontsize=14, fontweight='bold')
    plt.ylabel('True Labels', fontsize=14, fontweight='bold')
    plt.title("Confusion Matrix", fontsize = 16, fontweight = 'bold')
    for label in disp.ax_.get_xticklabels() + disp.ax_.get_yticklabels():
        label.set_fontsize(12)
        label.set_fontweight('bold')
    for text in disp.ax_.texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')
    plt.draw()
    return y_true, y_pred


def pca_cluster_all():
    df = get_dataframe_for_pca(True)
    df = df[df['SUBJECT'] != 'IDU037']
    matrix = df.drop('SUBJECT', axis = 1).to_numpy()
    matrix = scaler.fit_transform(matrix)
    subjects = df['SUBJECT']
    columns = df.columns[1:]
    pca = PCA()
    pca.fit(matrix)
    pca_result = pca.fit_transform(matrix)
    pca_result_2 = pca_result
    loadings = pca.components_
    n_cluster, kmeans = find_optimal_cluster(pca_result[:,:3], max_k = matrix.shape[0])
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    n_components = np.argmax(cumulative_variance >= 0.95) + 1
    loadings = pca.components_
    cluster_labels = kmeans.labels_
    df = get_dataframe_for_pca(True)
    # df = df[df['SUBJECT'] != 'IDU037']
    matrix = df.drop('SUBJECT', axis = 1).to_numpy()
    matrix = scaler.fit_transform(matrix)
    subjects = df['SUBJECT']
    columns = df.columns[1:]
    pca = PCA()
    pca.fit(matrix)
    pca_result = pca.fit_transform(matrix)
    loadings = pca.components_
    n_cluster, kmeans = find_optimal_cluster(pca_result[:,:3], max_k = matrix.shape[0])
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    n_components = np.argmax(cumulative_variance >= 0.95) + 1
    loadings = pca.components_
    cluster_labels_2 = np.insert(cluster_labels, -1, 2)
    colors = ['blue', 'green', 'orange']
    cluster_colors = [colors[label] for label in cluster_labels_2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection= '3d')
    ax.scatter(pca_result[:,0], pca_result[:,1], pca_result[:,2], c=cluster_colors, marker = 'o')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    
    # Optionally label each point with its subject
    for i, subject in enumerate(subjects):
        ax.text(pca_result[i, 0], pca_result[i, 1], pca_result[i, 2], subject, size=8)
    
    plt.title("PCA Cluster")
    plt.draw()
    fig = plt.figure()
    plt.scatter(pca_result[:,0], pca_result[:,1], c=cluster_colors, marker = 'o', linewidths=2)
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(12)
        label.set_fontweight('bold')
    # Optionally label each point with its subject
    for i, subject in enumerate(subjects):
        plt.text(pca_result[i, 0], pca_result[i, 1], subject, size=8, fontweight = 'bold')
    import matplotlib.lines as mlines
    blue_patch = mlines.Line2D([], [], color='blue', marker='o', linestyle=' ', markersize=8, label='Group 1')
    green_patch = mlines.Line2D([], [], color='green', marker='o', linestyle=' ', markersize=8, label='Group 2')
    purple_patch = mlines.Line2D([], [], color='orange', marker='o', linestyle=' ', markersize=8, label='Group 3')

    plt.legend(handles=[blue_patch, green_patch, purple_patch], fontsize=12, prop=font_manager.FontProperties(weight='bold') )
    plt.title("PCA: Subjects Clusters", fontsize=16, fontweight='bold')
    plt.xlabel('PC1', fontsize=14, fontweight='bold')
    plt.ylabel('PC2', fontsize=14, fontweight='bold')
    for i, subject in enumerate(subjects):
        plt.text(pca_result[i, 0], pca_result[i, 1], subject, size=8, fontweight='bold')
    plt.draw()
    df = {'SUBJECT' : subjects.to_list(),
          'LABEL' : cluster_labels}
    # print(df)
    # df = pd.DataFrame(df)
    # df.to_csv('stat/pca/label.csv', index=False)

    return pca_result, cluster_labels



if __name__ == '__main__':
    list_analysis = [(1, 0, 0, 0, 0),
                     (0, 1, 0, 0, 0),
                     (0, 0, 1, 0, 0),
                     (0, 0, 0, 1, 0),
                     (0, 0, 0, 0, 1),
                     (1, 1, 1, 1, 1),
                     (1, 1, 0, 0, 0),
                     (0, 0, 0, 1, 1)]  
    list_analysis = [[bool(value) for value in row] for row in list_analysis]
    import time
    # pca_result,cluster_labels = pca_cluster_all()
    # print(pca_result.shape)
    # pca_result = np.delete(np.asanyarray(pca_result), -2, axis=0)
    # print(pca_result.shape)
    # for bools in list_analysis:
    # bools = (True, True, False, False, False)
    bools = (True, True, False, False, False)
    # bools = (False, True, False, False, False)
    big_5, bis_11, gonogo, stroop_c, stroop_i = bools
    print(big_5, bis_11, gonogo, stroop_c, stroop_i)
    pca_result, subjects, cluster_labels, pca= pca(big_5, bis_11, gonogo, stroop_c, stroop_i)
    y_true, y_prd = svm()
    colors = ['blue', 'green', 'purple']
    colors = ['purple', 'orange']
    cluster_colors = [colors[label] for label in cluster_labels]
    fig = plt.figure()
    plt.scatter(pca_result[:,0], pca_result[:,1], c=cluster_colors, marker = 'o', linewidths=2)
    plt.title('Cluster Predicted - SVM with Errors', fontsize=16, fontweight='bold')
    plt.xlabel('PC1', fontsize=14, fontweight='bold')
    plt.ylabel('PC2', fontsize=14, fontweight='bold')
    for i, subject in enumerate(subjects):
        plt.text(pca_result[i, 0], pca_result[i, 1], subject, size=8, fontweight='bold')
    # incorrect_indices = np.where(y_true != y_prd)[0]
    pca_result_incorrect = []
    for idx in range(len(y_prd)):
        if not y_prd[idx] == y_true[idx]:
            pca_result_incorrect.append([pca_result[idx,0], pca_result[idx,1]])
    pca_result_incorrect = np.asanyarray(pca_result_incorrect)
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(12)
        label.set_fontweight('bold')
    # Use the indices to get the corresponding rows from pca_result
    # pca_result_incorrect = pca_result[incorrect_indices, :2]

    # Now you can plot the incorrect predictions
    # plt.scatter(pca_result_incorrect[:, 0], pca_result_incorrect[:, 1], c='red', marker='x')
    # pca_result_incorrect = [[pca_result[i,0], pca_result[i,1]] for i in range(pca_result.shape[0]) if not y_true[i] == y_prd[i]]
    plt.scatter(pca_result_incorrect[:,0], pca_result_incorrect[:,1], c='red', marker = 'x', linewidths=2)
 
    # import matplotlib.lines as mlines
    # green_patch = mlines.Line2D([], [], color='green', marker='o', linestyle=' ', markersize=8, label='Correct')
    # red_patch = mlines.Line2D([], [], color='red', marker='o', linestyle=' ', markersize=8, label='Incorrect')

    # plt.legend(handles=[green_patch, red_patch])
    # print(bools)
    # pca()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for predicted clusters
    ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=cluster_colors, marker='o', s=50, linewidths=1)

    # Titles and labels
    ax.set_title('Cluster Predicted - SVM with Errors', fontsize=16, fontweight='bold')
    ax.set_xlabel('PC1', fontsize=14, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=14, fontweight='bold')
    ax.set_zlabel('PC3', fontsize=14, fontweight='bold')

    # Add text labels for each subject
    # for i, subject in enumerate(subjects):
    #     ax.text(pca_result[i, 0], pca_result[i, 1], pca_result[i, 2], subject, size=10, fontweight='bold')

    # Identify incorrect predictions
    pca_result_incorrect = []
    for idx in range(len(y_prd)):
        if not y_prd[idx] == y_true[idx]:
            pca_result_incorrect.append([pca_result[idx, 0], pca_result[idx, 1], pca_result[idx, 2]])

    pca_result_incorrect = np.array(pca_result_incorrect)

    # Scatter plot for incorrect predictions in red
    ax.scatter(pca_result_incorrect[:, 0], pca_result_incorrect[:, 1], pca_result_incorrect[:, 2], c='red', s=150, marker='x', linewidths=2)

    # Set font size and weight for axis labels
    for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        label.set_fontsize(12)
        label.set_fontweight('bold')
    
    import matplotlib.lines as mlines
    purple = mlines.Line2D([], [], color='purple', marker='o', linestyle=' ', markersize=8, label='Group 1')
    orange = mlines.Line2D([], [], color='orange', marker='o', linestyle=' ', markersize=8, label='Group 2')
    crosses = mlines.Line2D([], [], color='red', marker='x', linestyle=' ', markersize=8, label='Errors')

    plt.legend(handles=[purple, orange, crosses], fontsize=12, prop=font_manager.FontProperties(weight='bold'))


    plt.show()