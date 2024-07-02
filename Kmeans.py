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

    def GetMeans(self,x):
        length = len(x)
        return [sum([element[0] for element in x])/length,sum([element[1] for element in x])/length, sum([element[2] for element in x])/length]

    def fit(self, x):
        """
        用fit方法对数据进行聚类
        :param x: 输入数据
        :best_centers: 簇中心点坐标 数据类型: ndarray
        :best_labels: 聚类标签 数据类型: ndarray
        :return: self
        """
        M = float('-inf')#maxscore
        best_labels = []
        best_centers = []
        for init_i in range(self.n_init):
            R = []#result
            C = []#center
            for i in range(self.n_clusters): 
                # random initialization
                Cnow =[random.uniform(x['Dimension1'].min(),
                             x['Dimension1'].max()),random.uniform(x['Dimension2'].min(),
                             x['Dimension2'].max()),random.uniform(x['Dimension3'].min(),x['Dimension3'].max())]
        C.append(Cnow)
        C = np.array(C)
        for iterrativetimes in range(self.max_iter):
            Ltemp =[]#labeltemp
            Ctemp = [] # centertemp
            for j in range(self.n_clusters):
                Ctemp.append([])
            for i in range(len(x)): 
                now = np.array(x.loc[i,0:3])
                Dmin = float('inf')#distance minimum
                Kmin = 0 
                for j in range(self.n_clusters):# find the nearest mean
                    now2 = np.array(C[j]) 
                    if np.linalg.norm(now, now2) < Dmin:
                        Dmin = np.linalg.norm(now, now2)
                        Kmin = j
                Ctemp[Kmin].append(now)
                Ltemp.append(Kmin)
                for i in range(self.n_clusters): # calculate the center
                    if len(Ctemp[i])-0: 
                        C[i] =[random.uniform(x['Dimension1'].min(),x['Dimension1'].max()),random.uniform(x['Dimension2'].min(),x['Dimension2'].max()),random.uniform(x['Dimension3'].min(),x['Dimension3'].max())]
                    # renew the center
                    else:
                        C[i] = self.GetMeans(Ctemp[i])
                        R = copy.deepcopy(tmp_abel)
                    Snow=silhouette_score(x,R) 
                    if Snow > M:
                       M = Snow
                       best_labels = copy.deepcopy(R)
                       best_centers = copy.deepcopy(C)
                self.cluster_centers_ = best_centers
                self.labels_ = best_labels
                return self
    
    
