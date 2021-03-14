#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2021/3/13 0013 23:03
# @Author&Email: COLFLIP&colflip@163.com
# @File: Algorithms1.py
# @Software: PyCharm

# ---------------------------------------------
# # Algorithms1：unnormalized Spectral Clustering
# kmeans
# ---------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans


def getDistanceMatrix(data):
    """
    获取距离矩阵
    :param data: 样本集合
    :return: 距离矩阵
    """
    n = len(data)  # 样本总数
    dist_matrix = np.zeros((n, n))  # 初始化矩阵为n×n的全0矩阵
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.power(data[i] - data[j], 2).sum())
            dist_matrix[i][j] = dist_matrix[j][i] = dist
    return dist_matrix


def getAdjacencyMatrix(data, k):
    """
    获得邻接矩阵AdjacencyMatrix W
    :param data: 样本集合
    :param k : K参数
    :return: W
    """
    n = len(data)
    dist_matrix = getDistanceMatrix(data)
    W = np.zeros((n, n))
    for idx, item in enumerate(dist_matrix):
        idx_array = np.argsort(item)  # 每一行距离列表进行排序,得到对应的索引列表
        W[idx][idx_array[1:k + 1]] = 1
    transpW = np.transpose(W)
    return (W + transpW) / 2


def getDegreeMatrix(W):
    """
    获得度矩阵Degree
    :param W: 邻接矩阵
    :return: D
    """
    D = np.diag(sum(W))
    return D


def getLaplacianMatrix(D, W):
    """
    获得拉普拉斯矩阵
    :param W: 邻接矩阵
    :param D: 度矩阵
    :return: L
    """
    return D - W


def getEigen(L, cluster_num):
    """
    获得拉普拉斯矩阵的特征矩阵
    :param L:
    :param cluter_num: 聚类数目
    :return:
    """
    eigval, eigvec = np.linalg.eig(L)
    ix = np.argsort(eigval)[0:cluster_num]
    return eigvec[:, ix]


def plotRes(data, clusterResult, clusterNum):
    """
    结果可似化
    :param data:  样本集
    :param clusterResult: 聚类结果
    :param clusterNum: 聚类个数
    :return:
    """
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


def USC(data, cluster_num, k):
    W = getAdjacencyMatrix(data, k)
    D = getDegreeMatrix(W)
    L = getLaplacianMatrix(D, W)
    # print(L)

    eigvec = getEigen(L, cluster_num)
    clf = KMeans(n_clusters=cluster_num)
    s = clf.fit(eigvec)  # 聚类
    label = s.labels_
    return label


cluster_num = 7
knn_k = 5
filename = 'Aggregation_cluster=7.txt'
data = np.loadtxt(filename, delimiter='\t')
data = data[0:-1]  # 除了最后一列 最后一列为标签列
data = np.array(data)
# plt.scatter(data[:, 0], data[:, 1], marker='+')
# plt.show()
label = USC(data, cluster_num, knn_k)
plotRes(data, label, cluster_num)

print(metrics.silhouette_score(data, label))  # 轮廓系数评价
print(metrics.davies_bouldin_score(data, label))  # 戴维森堡丁指数(DBI)评价
print(metrics.calinski_harabasz_score(data, label))  # CH指标评价
