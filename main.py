import os

import pickle
from joblib import dump, load
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as ss
import numpy as np
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from os.path import isfile
import subprocess
import streamlit as st


def calc_common_usrs(data_dir, pod1, pod2):
    pod_list1 = [line.rstrip('\n') for line in open(data_dir + pod1)]
    pod_list1.pop(0)
    pod_list2 = [line.rstrip('\n') for line in open(data_dir + pod2)]
    pod_list2.pop(0)
    num_same = set(pod_list1) & set(pod_list2)
    return len(num_same), len(pod_list1), len(pod_list2)


cluster_pick = st.sidebar.selectbox(
    'Which number do you like best?',
    [2, 3, 4, 5, 6])

data_dir = '/home/ed/github/pod_tweets/follower_ids/'
resources_dir = '/home/ed/github/pod_tweets/resources/'
pod_list = os.listdir(data_dir)
num_pods = len(pod_list)

pod_common_usrs = pd.DataFrame(columns=['podcast_1', 'podcast_2', 'comm_users'])
pod_popularity = pd.DataFrame(pod_list, columns=['podcast'])
pod_popularity['followers'] = 0
pod_popularity = pod_popularity.set_index('podcast')

for x1 in range(num_pods):
    for x2 in range(x1+1, num_pods):
        comm_num, pod_x1, pod_x2 = calc_common_usrs(data_dir, pod_list[x1], pod_list[x2])
        pod_common_usrs.loc[len(pod_common_usrs)] = [pod_list[x1], pod_list[x2], comm_num]
        pod_popularity.at[pod_list[x1], 'followers'] = pod_x1
        pod_popularity.at[pod_list[x2], 'followers'] = pod_x2
print(pod_popularity)

pod_popularity = pod_popularity.sort_values(by='followers', ascending=False)
pod_pop_arr = np.array(pod_popularity.index)

index_map = dict(np.vstack([pod_pop_arr, np.arange(pod_pop_arr.shape[0])]).T)

count_matrix = ss.coo_matrix((pod_common_usrs.comm_users,
                              (pod_common_usrs.podcast_2.map(index_map),
                               pod_common_usrs.podcast_1.map(index_map))),
                             shape=(pod_pop_arr.shape[0], pod_pop_arr.shape[0]),
                             dtype=np.float64)

conditional_prob_matrix = count_matrix.tocsr()
conditional_prob_matrix = normalize(conditional_prob_matrix, norm='l1', copy=False)

reduced_data = TruncatedSVD(n_components=2).fit_transform(conditional_prob_matrix)
kmeans = KMeans(init='k-means++', n_clusters=cluster_pick, n_init=10)
kmeans.fit(reduced_data)

data_dump = resources_dir + '20193101_data.joblib'
dump(reduced_data, data_dump)

model_dump = resources_dir + '20193101_kmeans.joblib'
dump(kmeans, model_dump)

# Plot
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on PCA-reduced data\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
# plt.show()
st.pyplot()
