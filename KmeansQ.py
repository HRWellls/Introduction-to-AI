import os
import sklearn
import numpy as np
import pandas as pd
import random
from copy import deepcopy
from sklearn.externals import joblib
# import joblib
import numpy as np

class KMeans():
    """
    Parameters
    ----------
    n_clusters 指定了需要聚类的个数，这个超参数需要自己调整，会影响聚类的效果
    n_init 指定计算次数，算法并不会运行一遍后就返回结果，而是运行多次后返回最好的一次结果，n_init即指明运行的次数
    max_iter 指定单次运行中最大的迭代次数，超过当前迭代次数即停止运行
    """
    def __init__(
                self,
                n_clusters=8,
                n_init=10,
                max_iter=300
                ):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
    def GetDistance(self, x, y):
        """
        计算两点之间的距离
        """
        return np.linalg.norm(x-y)
    def GetMeans(self,x):
        length = len(x)
        return [sum([element[0] for element in x])/length,
        sum([element[1] for element in x])/length, sum([element[2] for element in x])/length]
    def fit(self, x):
        """
        用fit方法对数据进行聚类
        :param x: 输入数据
        :best_centers: 簇中心点坐标 数据类型: ndarray
        :best_labels: 聚类标签 数据类型: ndarray
        :return: self
        """
        max_score = float('-inf')
        best_labels = []
        best_centers = []
        for init_i in range(self.n_init):
            results = []
            centers = []
            for i in range(self.n_clusters): 
                # 随机初始化聚类
                this_center =[random.uniform(x['Dimension1'].min(),
                             x['Dimension1'].max()),random.uniform(x['Dimension2'].min(),
                             x['Dimension2'].max()),random.uniform(x['Dimension3'].min(),x['Dimension3'].max())]
        centers.append(this_center)
        centers = np.array(centers)# 便于计算距离
        for iter_ in range(self.max_iter):
            tmp_label =[]
            tmp_centers = [] # 二维数组，每一个元素是对应聚类所包含的数据点
            for j in range(self.n_clusters):
                tmp_centers.append([])
            for i in range(len(x)): # 遍历每一个数据点
                now = np.array(x.loc[i,0:3])
                min_dis = float('inf')
                min_k = 0 
                for j in range(self.n_clusters):# 找这个点距离最近的聚类
                    now2 = np.array(centers[j]) #以便利用 getdistance 比较
                    if self.GetDistance(now, now2) < min_dis:
                        min_dis = self.GetDistance(now, now2)
                        min_k = j
                tmp_centers[min_k].append(now)
                tmp_label.append(min_k)
                for i in range(self.n_clusters): # 计算每一个聚类中点的中心值
                    if len(tmp_centers[i])-0: # 如果聚类为空，则直接再次随机一个点
                        centers[i] =[random.uniform(x['Dimension1'].min(),x['Dimension1'].max()),random.uniform(x['Dimension2'].min(),x['Dimension2'].max()),random.uniform(x['Dimension3'].min(),x['Dimension3'].max())]
                    # 更新聚类中心的坐标
                    else:
                        centers[i] = self.GetMeans(tmp_centers[i])
                        results = copy.deepcopy(tmp_abel)
                    now_score=silhouette_score(x,results) # 给这个聚类评分
                    if now_score > max_score:
                       max_score = now_score
                       best_labels = copy.deepcopy(results)
                       best_centers = copy.deepcopy(centers)
                self.cluster_centers_ = best_centers
                self.labels_ = best_labels
                return self
                

def preprocess_data(df):
    """
    数据处理及特征工程等
    :param df: 读取原始 csv 数据，有 timestamp、cpc、cpm 共 3 列特征
    :return: 处理后的数据, 返回 pca 降维后的特征
    """
    # ====================数据预处理、构造特征等========================
    # 例如
    # df['hours'] = df['timestamp'].dt.hour
    # df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)

    # ========================  模型加载  ===========================
    # 请确认需要用到的列名，e.g.:columns = ['cpc','cpm']
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    df['cpc X cpm'] = df['cpm'] * df['cpc']
    df['cpc / cpm']= df['cpc'] / df['cpm']
    df['hours'] = df['timestamp'].dt.hour
    df['daylight'] = ((df['hours'] >= 7)& (df['hours'] <= 22)).astype(int)
    # 请使用joblib函数加载自己训练的 scaler、pca 模型，方便在测试时系统对数据进行相同的变换
    scaler = joblib.load('./results/scaler.pkl')
    pca = joblib.load('./results/pca.pkl')
    columns =['cpc','cpm','cpc X cpm','cpc / cpm']
    data = df[columns]
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data, columns=columns)
    data = pca.fit_transform(data)
    data = pd.DataFrame(data,columns=['Dimension' + str(i+1) for i in range(3)])

    # 例如
    # scaler = joblib.load('./results/scaler.pkl')
    # pca = joblib.load('./results/pca.pkl')
    # data = scaler.transform(data)

    return data

def get_distance(data, kmeans, n_features):
    """
    计算样本点与聚类中心的距离
    :param data: preprocess_data 函数返回值，即 pca 降维后的数据
    :param kmeans: 通过 joblib 加载的模型对象，或者训练好的 kmeans 模型
    :param n_features: 计算距离需要的特征的数量
    :return:每个点距离自己簇中心的距离，Series 类型
    """
    # ====================计算样本点与聚类中心的距离========================
    distance = []
    for i in range(len(data)):
        point = np.array(data.iloc[i,:n_features])
        center = kmeans.cluster_centers_[kmeans.abels_[i]]
        distance.append(np.linalg.norm(point - center))
    distance = pd.Series(distance)
    return distance

def get_anomaly(data, kmeans, ratio):
    """
    检验出样本中的异常点，并标记为 True 和 False，True 表示是异常点
    
    :param data: preprocess_data 函数返回值，即 pca 降维后的数据，DataFrame 类型
    :param kmeans: 通过 joblib 加载的模型对象，或者训练好的 kmeans 模型
    :param ratio: 异常数据占全部数据的百分比,在 0 - 1 之间，float 类型
    :return: data 添加 is_anomaly 列，该列数据是根据阈值距离大小判断每个点是否是异常值，元素值为 False 和 True
    """
    # ====================检验出样本中的异常点========================
    num_anomaly = int(en(data)* ratio)
    new_data = deepcopy(data)
    new_data['distance'] = get_distance(new_data,kmean,n_features=en(new_data.columns))
    threshould = new_data['distance'].sort_values(ascending=False).reset_index(drop=True)[num_anomaly]
    new_data['is_anomaly'] = new_data['distance'].apply(lambda x:x>threshould)
    normal = new_data[new_data['is_anomaly'] == 0]
    anormal = new_data[new_data['is_anomaly'] == 1]
    return new_data