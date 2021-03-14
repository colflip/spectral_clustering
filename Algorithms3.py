#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2021/3/14 0014 15:53
# @Author&Email: COLFLIP&colflip@163.com
# @File: Algorithms3.py
# @Software: PyCharm

# ---------------------------------------------
# # Algorithms3：Normalized spectral clustering according toShi and Malik(2000)
# Lsym:=D−1/2LD−1/2=I−D−1/2WD−1/2
# Lrw:=D−1L=I−D−1W
# ---------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import fractional_matrix_power
from sklearn import metrics
from sklearn.cluster import KMeans


def getDistanceMatrix(data):
    n = len(data)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.power(data[i] - data[j], 2).sum())
            dist_matrix[i][j] = dist_matrix[j][i] = dist
    return dist_matrix


def getAdjacencyMatrix(data):
    k = 4
    n = len(data)
    dist_matrix = getDistanceMatrix(data)
    W = np.zeros((n, n))
    for idx, item in enumerate(dist_matrix):
        idx_array = np.argsort(item)
        W[idx][idx_array[1:k + 1]] = 1
    transpW = np.transpose(W)
    return (W + transpW) / 2


def getDegreeMatrix(W):
    D = np.diag(sum(W))
    return D


def getLaplacianMatrix(D, W):
    return D - W


def getEigen(L, D, k):
    """
    获得归一化拉普拉斯矩阵的特征矩阵 Lsym
    :param L:
    :param cluter_num: 聚类数目
    :return:
    """
    DD = fractional_matrix_power(D, -0.5)
    L = DD @ L @ DD
    # print(L)
    eigval, eigvec = np.linalg.eig(L)
    ix = np.argsort(eigval)[0:k]
    return eigvec[:, ix]


def plotRes(data, clusterResult, clusterNum):
    n = len(data)
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'LightGrey']
    for i in range(clusterNum):
        color = scatterColors[i % len(scatterColors)]
        x1 = [];
        y1 = []
        for j in range(n):
            if clusterResult[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
        plt.scatter(x1, y1, c=color, marker='+')
    plt.show()


def NSCByNg(data, k):
    W = getAdjacencyMatrix(data)
    D = getDegreeMatrix(W)
    L = getLaplacianMatrix(D, W)
    eigvec = getEigen(L, D, k)

    # tij = uij / (∑ku2ik)    1 / 2
    rows = eigvec.shape[0]
    columns = eigvec.shape[1]
    T = np.zeros((rows, columns))
    for i in range(rows):
        for j in range(columns):
            T[i][j] = eigvec[i][j] / np.sqrt(np.sum(eigvec[i] ** 2))
    clf = KMeans(n_clusters=k)
    s = clf.fit(T)
    label = s.labels_
    return label


k = 7
filename = 'Aggregation_cluster=7.txt'
data = np.loadtxt(filename, delimiter='\t')
data = data[0:-1]  # 除了最后一列 最后一列为标签列
data = np.array(data)
label = NSCByNg(data, k)
plotRes(data, label, k)

print(metrics.silhouette_score(data, label))  # 轮廓系数评价
print(metrics.davies_bouldin_score(data, label))  # 戴维森堡丁指数(DBI)评价
print(metrics.calinski_harabasz_score(data, label))  # CH指标评价
