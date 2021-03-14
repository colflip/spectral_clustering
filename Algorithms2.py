#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2021/3/13 0013 23:21
# @Author&Email: COLFLIP&colflip@163.com
# @File: Algorithms2.py
# @Software: PyCharm

# ---------------------------------------------
# # Algorithms2：Normalized spectral clustering according toShi and Malik(2000)
# generalized eigenproblem Lu=λDu
# ---------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
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
    k = 5
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
    获得广义拉普拉斯矩阵的特征矩阵 Lu=λDu
    :param L:
    :param cluter_num: 聚类数目
    :return:
    """
    DD = np.linalg.inv(D)
    L = DD @ L
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


def NSCBySAndM(data, k):
    W = getAdjacencyMatrix(data)
    D = getDegreeMatrix(W)
    L = getLaplacianMatrix(D, W)
    eigvec = getEigen(L, D, k)
    clf = KMeans(n_clusters=k)
    s = clf.fit(eigvec)
    label = s.labels_
    return label


k = 7
filename = 'Aggregation_cluster=7.txt'
data = np.loadtxt(filename, delimiter='\t')
data = data[0:-1]  # 除了最后一列 最后一列为标签列
data = np.array(data)
label = NSCBySAndM(data, k)
plotRes(data, label, k)

print(metrics.silhouette_score(data, label))  # 轮廓系数评价
print(metrics.davies_bouldin_score(data, label))  # 戴维森堡丁指数(DBI)评价
print(metrics.calinski_harabasz_score(data, label))  # CH指标评价
