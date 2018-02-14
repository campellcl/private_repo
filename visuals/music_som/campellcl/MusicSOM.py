"""
MusicSOM.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


__author__ = 'Chris Campell'
__version__ = '2/11/2018'

with open('../../../data/chords.csv', 'r') as fp:
    chords = pd.read_csv(fp)
df_chords = pd.DataFrame(chords)

# Define the map size to be 20 on each axis:
p = 20
# Define the number of attributes to be everything but the target label:
m = len(df_chords.columns) - 1
# Define the iteration limit (number of epochs?) to be 360:
lambda_iter = 360
# lambda_iter = 50
# Define the number of samples n:
n = df_chords.shape[0]
# Define c, the neighborhood radius decrement period
c = 20
# Define alpha, the learning rate
alpha = 0.02
# Initialize a weight matrix with the same shape as the input matrix
weights = np.zeros(shape=(p, p, m))


def custom_dist(i, j, k, l):
    """
    custom_dist: A custom distance function handling the torus shape of the neuron grid that is superimposed on the
        lower dimensional subspace.
    :param i: The first index into the 3D array of chords.
    :param j: The second index into the 3D array of chords.
    :param k: The first index into the 3D array of nodes.
    :param l: The second index into the 3D array of nodes.
    :return distance: The euclidean distance between
    """
    dx = np.abs(i - k)
    if dx > 9:
        # Wrap around torus:
        dx = p - dx
    dy = np.abs(j - l)
    if dy > 9:
        # Wrap around torus:
        dy = p - dy
    euclidean_distance = np.sqrt((dx**2) + (dy**2))
    return euclidean_distance

# Initialize weight vector utilizing normal distribution:
# TODO: More efficient way to iterate this structure?
for i in range(p):
    for j in range(p):
        for k in range(m):
            weights[i, j, k] = np.random.uniform(0, 1)

best_matching_units = {}


# Perform the training over lambda iterations.
for s in range(0, lambda_iter):
    # Randomly pick an input vector (sample with replacement)
    for t in range(0, n):
        # Select the values of a random input vector
        random_index = np.random.randint(0, n)
        r = df_chords.iloc[random_index][:].values
        # This is the chord associated with the input vector:
        label_t = r[0]
        # r_t is a vector of size 12 (randomly selected)
        r_t = r[1:]
        # These two vars will hold the index of the BMU for the randomly selected vector.
        t_i, t_j = (-1, -1)
        min_dist = None
        BMU = None
        # Traverse each node in the map.
        for i in range(p):
            for j in range(p):
                '''
                Use Euclidean distance formula to find the similarity between the input vector and map's node's weight
                    vector. Keep track of the node that produces the smallest distance, call it the BMU. 
                '''
                if i == 0 and j == 0:
                    current_dist = np.linalg.norm(weights[0, 0] - r_t)
                    min_dist = current_dist
                    BMU = weights[0, 0]
                    t_i, t_j = (0, 0)
                else:
                    current_dist = np.linalg.norm(weights[i, j] - r_t)
                    if current_dist < min_dist:
                        min_dist = current_dist
                        BMU = weights[i, j]
                        t_i, t_j = (i, j)
        '''
        Update the weight vectors of nodes in the neighborhood of the BMU.
        '''
        for k in range(p):
            for l in range(p):
                sigma = (1/3) * (p - 1 - (s/c))
                theta = np.exp(-((custom_dist(t_i, t_j, k, l)**2)/(2*(sigma**2))))
                weights[k, l] = weights[k, l] + (theta*alpha)*(r_t - weights[k, l])
                        # print(weights)
        # Keep track of the best matching unit indices
        best_matching_units[label_t] = (t_i, t_j)
        # print("The BMU for randomly selected chord %s is at index (%d,%d)." % (label_t, t_i, t_j))
    print('Ended iteration %d.' % s)

ax, fig = plt.subplots()
plt.xlim(0, 20)
plt.xticks(np.arange(0, 21, 1.0))
plt.ylim(0, 20)
plt.yticks(np.arange(0, 21, 1.0))
plt.grid()
x_coords = [bmu_index[0] for (label, bmu_index) in best_matching_units.items()]
y_coords = [bmu_index[1] for (label, bmu_index) in best_matching_units.items()]
labels = [label for (label, bmu_index) in best_matching_units.items()]
plt.scatter(x=x_coords, y=y_coords)
annotated = []
for i, label in enumerate(labels):
    if annotated:
        if label not in annotated:
            plt.annotate(label, (x_coords[i], y_coords[i]))
            annotated.append(label)
    else:
        plt.annotate(label, (x_coords[i], y_coords[i]))
        annotated.append(label)
plt.savefig('music_som.png')
plt.show()


# Colormap: Take CM and dot product it with every W_ij
CM = df_chords[df_chords['Label'] == 'CM'].values[0]
CM = CM[1:]
CM_activation = np.zeros((20, 20))
for i in range(p):
    for j in range(p):
        CM_activation[i,j] = (np.dot(weights[i, j], CM))
plt.clf()
ax, fig = plt.subplots()
plt.xlim(0, 19)
plt.xticks(np.arange(0, 20, 1.0))
plt.ylim(0, 19)
plt.yticks(np.arange(0, 20, 1.0))
# sns.heatmap(CM_activation, annot=False,  linewidths=.5, vmin=0, vmax=1)
sns.heatmap(CM_activation, annot=False, linewidths=.5)
plt.savefig('music_som_CM.png')

cm = df_chords[df_chords['Label'] == 'Cm'].values[0]
cm = cm[1:]
cm_activation = np.zeros((20, 20))
for i in range(p):
    for j in range(p):
        cm_activation[i,j] = (np.dot(weights[i, j], cm))
plt.clf()
ax, fig = plt.subplots()
plt.xlim(0, 19)
plt.xticks(np.arange(0, 20, 1.0))
plt.ylim(0, 19)
plt.yticks(np.arange(0, 20, 1.0))
sns.heatmap(CM_activation, annot=False, linewidths=.5)
plt.savefig('music_som_Cm.png')

