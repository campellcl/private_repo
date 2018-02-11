"""
MusicSOM.py
"""

import pandas as pd
import numpy as np

__author__ = 'Chris Campell'
__version__ = '2/11/2018'

with open('../../../data/chords.csv', 'r') as fp:
    chords = pd.read_csv(fp)
df_chords = pd.DataFrame(chords)

def custom_dist(i, j, k, l):
    """
    custom_dist: A custom distance function handling the torus shape of the neuron grid that is superimposed on the
        lower dimensional subspace.
    :param i: The first index into the 3D array of chords.
    :param j: The second index into the 3D array of chords.
    :param k: The third index into the 3D array of chords (attribute selector).
    :param l: TODO: Unknown parameter!!!
    :return distance: The euclidean distance between
    """

# Define the map size to be 20 on each axis:
p = 20
# Define the number of attributes to be everything but the target label:
m = len(df_chords.columns) - 1
# Define the iteration limit (number of epochs?) to be 360:
lambda_iter = 360
# Define the number of samples n:
n = df_chords.shape[0]
# Define c, the neighborhood radius decrement period
c = 20
# Initialize a weight matrix with the same shape as the input matrix
weights = np.zeros(shape=(p, p, m))

# for i, j, k in zip(range(p),range(p), range(m)):
#     print(i, j, k)

# Initialize weight vector utilizing normal distribution:
# TODO: More efficent way to iterate this structure?
for i in range(p):
    for j in range(p):
        for k in range(m):
            weights[i, j, k] = np.random.uniform(0, 1)

for s in range(0, lambda_iter):
    for t in range(0, n):
        # Select the values of a random input vector
        r_t = df_chords.iloc[np.random.randint(0, n)][1:].values
        t_i, t_j = (-1, -1)
        BMU = None
        for i in range(p):
            for j in range(p):
                if i == 0 and j == 0:
                    update = np.abs(weights[i, j] - r_t)
                    BMU = update
                    t_i, t_j = (0, 0)
                else:
                    current_match = np.abs(weights[i, j] - r_t)
                    if sum(current_match) < sum(BMU):
                        BMU = current_match
                        t_i, t_j = (i, j)
                for k in range(p):
                    for l in range(p):
                        sigma = (1/3) * (p - 1 - (s/c))
                        theta = np.exp(-((custom_dist(i, j, k, l))/(2*(sigma**2))))


pass

# for i, j in zip(range(p), range(p)):
#     print(i, j)
#     for k in range(1,m):
#         weights[i, j, k] = np.random.uniform(0, 1)
#         print("k:%d" % k)


pass
