import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class SelfOrganizingMap():
    
    def __init__(self, input_dim, grid_size, lr = 0.5, radius = None, decay_rate = 0.99):
        self.input_dim = input_dim
        self.grid_size = grid_size
        self.lr = lr
        self.radius = radius or max(grid_size) / 2
        self.decay_rate = decay_rate
        self.weights = np.random.random((grid_size[0], grid_size[1], input_dim))
    
    def train(self, data, num_epochs):
        from tqdm import tqdm
        for epoch in tqdm(range(num_epochs)):
            for sample in data:
                bmu_idx = self.find_bmu(sample)
                self.update_weights(sample, bmu_idx, epoch, num_epochs)
            self.lr *= self.decay_rate
            self.radius *= self.decay_rate

    def find_bmu(self, sample):
        distances = np.linalg.norm(self.weights - sample, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), self.grid_size)
        return bmu_idx
    
    def update_weights(self, sample, bmu_idx, epoch, num_epochs):
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                # Calculate the Euclidean distance between the current node and the BMU
                dist_to_bmu = np.linalg.norm(np.array([x, y]) - np.array(bmu_idx))
                
                if dist_to_bmu <= self.radius:
                    influence = np.exp(-dist_to_bmu**2 / (2 * (self.radius ** 2)))
                    self.weights[x, y] += self.lr * influence * (sample - self.weights[x, y])

    def assign_clusters(self, data):
        labels = []
        for sample in data:
            bmu_idx = self.find_bmu(sample)
            labels.append(bmu_idx[0]*self.grid_size[1] + bmu_idx[1])
        return np.array(labels)
    
    def visualize(self):
        """
        Visualizes the SOM grid.
        """
        fig = plt.figure()
        plt.imshow(self.weights.reshape(self.grid_size[0], self.grid_size[1], -1))
        plt.title(f"Self-Organizing Map {self.grid_size[0]}x{self.grid_size[1]}")
        plt.draw()

    def visualize_2(self):
        """
        Visualizes the SOM grid.
        The weights are reshaped to be displayed as a 2D grid.
        """
        # Reshape the weights to display each weight vector in a 2D grid
        reshaped_weights = self.weights.reshape(self.grid_size[0], self.grid_size[1], -1)
        
        # If you want to visualize the SOM in a color format for each component of the weight vector:
        fig = plt.figure(figsize=(10, 10))

        # Visualizing the first principal component (you can choose any component, or show them all)
        # Here, we are visualizing the first component (0th dimension) of the weight vectors
        plt.imshow(reshaped_weights[:, :, 0], cmap='viridis')
        plt.colorbar()
        plt.title("Self-Organizing Map: Visualizing First Component")
        plt.draw()

        # Optionally, you could visualize the other components (e.g., PC1, PC2, etc.)
        for i in range(reshaped_weights.shape[2]):  # Iterate over all components
            fig = plt.figure()
            plt.imshow(reshaped_weights[:, :, i], cmap='viridis')
            plt.colorbar()
            plt.title(f"Self-Organizing Map: Visualizing Component {i+1}")
            plt.draw()
    
if __name__ == '__main__':

    from pca_cluster_personality import pca
    from sklearn.metrics import silhouette_score

    bools =(True, True, False, False, False)
    bools = tuple(True for _ in range(5))
    big_5, bis_11, gonogo, stroop_c, stroop_i = bools
    ignore_some_subjects = True
    pca_result, subjects, cluster_labels, pca= pca(big_5, bis_11, gonogo, stroop_c, stroop_i, ignore_some_subjects)
    silhouette_scores = []
    som_list = []

    # Define SOM grid sizes (similar to varying `k` in K-means)
    grid_sizes = [(2, 2), (3, 3), (4, 4)]  
    grid_sizes = [(1,2), (2,1), (2,2), (4,4),(10,10),(100, 100),(200,200)]
    print(pca_result.shape)

    # for grid_size in grid_sizes:
    from tqdm import tqdm
    for grid_size in tqdm(grid_sizes):
        for j in range(3):
            for k in range(2):
                    if k == 0:
            # Initialize and train the SOM
                       som = SelfOrganizingMap(input_dim=pca_result.shape[1], grid_size=grid_size, lr=0.1, radius= max(grid_size))
                    else:
                        som = SelfOrganizingMap(input_dim=pca_result.shape[1], grid_size=grid_size, lr=0.1)
                    som.train(pca_result, num_epochs=250)
                    # som.visualize_2()
                    
                    # Assign clusters
                    cluster_labels = som.assign_clusters(pca_result)
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection= '3d')
                    cmap = plt.cm.jet
                    unique_labels = np.unique(cluster_labels)
                    norm = plt.Normalize(vmin=min(unique_labels), vmax=max(unique_labels))
                    print(unique_labels)
                    # Scatter plot with cluster labels (colors)
                    colors = [cmap(norm(label)) for label in cluster_labels]
                    ax.scatter(pca_result[:,0], pca_result[:,1], pca_result[:,2], c=colors, marker = 'o')
                    ax.set_xlabel('PC1')
                    ax.set_ylabel('PC2')
                    ax.set_zlabel('PC3')
                    
                    # Optionally label each point with its subject
                    for i, subject in enumerate(subjects):
                        ax.text(pca_result[i, 0], pca_result[i, 1], pca_result[i, 2], subject, size=8)
                    plt.title(f'SOM with Grid Size {grid_size[0]}x{grid_size[1]}')
                    # unique_labels = np.unique(cluster_labels)
                    handles = []
                    for idx, label in enumerate(unique_labels):
                        color = cmap(norm(label))
                        # Create a dummy plot entry for each unique label
                        handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,# markerfacecolor=plt.cm.jet(label / len(unique_labels)), 
                                            markersize=10, label=f'Cluster {label}')
                        handles.append(handle)
                    ax.legend(handles=handles, title="Clusters")

                    plt.draw()
                    plt.close('all')
                    fig = plt.figure()
                    activation_map = np.zeros(grid_size)
                    for label in cluster_labels:
                        x, y = divmod(label, grid_size[1])  # Convert flat index to 2D grid coordinates
                        activation_map[x, y] += 1

                    # Plot the heatmap
                    plt.figure(figsize=(8, 8))
                    plt.imshow(activation_map, cmap='hot', vmin=0,vmax = 16 if grid_size[0] < 4 else None, interpolation='nearest')
                    plt.colorbar(label='Number of Subjects')
                    plt.title("Heatmap of Activated Neurons")
                    plt.xlabel("Neuron X")
                    plt.ylabel("Neuron Y")
                    plt.draw()
                    # plt.savefig(f'stat/som/{grid_size[0]}x{grid_size[1]}_allFeatures_{j+1}.png')
                    if k== 0:
                        plt.savefig(f'stat/som/{grid_size[0]}x{grid_size[1]}_allFeatures_maxradius_{j+1}.png')
                    else:
                        plt.savefig(f'stat/som/{grid_size[0]}x{grid_size[1]}_allFeatures_{j+1}.png')
        # Compute silhouette score
        # score = silhouette_score(pca_result, cluster_labels)
        
        # som_list.append(som)
        # silhouette_scores.append(score)

    # Visualize silhouette scores
    # grid_sizes_str = [f"{size[0]}x{size[1]}" for size in grid_sizes]
    # # fig = plt.figure()
    # # plt.plot(grid_sizes_str, silhouette_scores, marker="o")
    # # plt.title("Silhouette Scores for Different SOM Grid Sizes")
    # # plt.xlabel("SOM Grid Size")
    # # plt.ylabel("Silhouette Score")
    # df = {'SUBJECT' : subjects.to_list(),
    #     'LABEL' : cluster_labels}
    # # print(df)
    # df = pd.DataFrame(df)
    # df.to_csv('stat/pca/label.csv', index=False)
    # from pca_cluster_personality import svm
    # y_true, y_prd = svm()
    # fig = plt.figure()
    # plt.scatter(pca_result[:,0], pca_result[:,1], c=colors, marker = 'o', linewidths=2)
    # plt.title('Cluster Predicted - SVM with Errors', fontsize=16, fontweight='bold')
    # plt.xlabel('PC1', fontsize=14, fontweight='bold')
    # plt.ylabel('PC2', fontsize=14, fontweight='bold')
    # for i, subject in enumerate(subjects):
    #     plt.text(pca_result[i, 0], pca_result[i, 1], subject, size=8, fontweight='bold')
    # # incorrect_indices = np.where(y_true != y_prd)[0]
    # pca_result_incorrect = []
    # for idx in range(len(y_prd)):
    #     if not y_prd[idx] == y_true[idx]:
    #         pca_result_incorrect.append([pca_result[idx,0], pca_result[idx,1]])
    # pca_result_incorrect = np.asanyarray(pca_result_incorrect)
    # ax = plt.gca()
    # for label in ax.get_xticklabels() + ax.get_yticklabels():
    #     label.set_fontsize(12)
    #     label.set_fontweight('bold')
    # # Use the indices to get the corresponding rows from pca_result
    # # pca_result_incorrect = pca_result[incorrect_indices, :2]

    # # Now you can plot the incorrect predictions
    # # plt.scatter(pca_result_incorrect[:, 0], pca_result_incorrect[:, 1], c='red', marker='x')
    # # pca_result_incorrect = [[pca_result[i,0], pca_result[i,1]] for i in range(pca_result.shape[0]) if not y_true[i] == y_prd[i]]
    # plt.scatter(pca_result_incorrect[:,0], pca_result_incorrect[:,1], c='red', marker = 'x', linewidths=2)
 
    # plt.show()