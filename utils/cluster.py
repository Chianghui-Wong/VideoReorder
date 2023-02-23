import torch
import random
import copy


class K_means():
    def __init__(self, data, k):
        self.data = data
        self.k = k

    def distance(self, p1, p2):
        return torch.sum((p1-p2)**2).sqrt()

    def generate_center(self):
        # 随机初始化聚类中心
        n = self.data.size(0)
        rand_id = random.sample(range(n), self.k)
        center = []
        for id in rand_id:
            center.append(self.data[id])
        return center

    def converge(self, old_center, new_center):
        # 判断是否收敛
        set1 = set(old_center)
        set2 = set(new_center)
        return set1 == set2

    def forward(self):
        center = self.generate_center()
        n = self.data.size(0)
        labels = torch.zeros(n).long()
        flag = False
        while not flag:
            old_center = copy.deepcopy(center)

            for i in range(n):
                cur = self.data[i]
                min_dis = 10*9
                for j in range(self.k):
                    dis = self.distance(cur, center[j])
                    if dis < min_dis:
                        min_dis = dis
                        labels[i] = j

            # 更新聚类中心
            for j in range(self.k):
                center[j] = torch.mean(self.data[labels == j], dim=0)

            flag = self.converge(old_center, center)

        return labels, center




def kmeans(x, ncluster, niter=10):
    '''
    x : torch.tensor(data_num,data_dim)
    ncluster : The number of clustering for data_num
    niter : Number of iterations for kmeans
    '''
    N, D = x.size()
    c = x[torch.randperm(N)[:ncluster]] # init clusters at random
    for i in range(niter):
        # assign all pixels to the closest codebook element
        # .argmin(1) : 按列取最小值的下标,下面这行的意思是将x.size(0)个数据点归类到random选出的ncluster类
        a = ((x[:, None, :] - c[None, :, :])**2).sum(-1).argmin(1)
        # move each codebook element to be the mean of the pixels that assigned to it
        # 计算每一类的迭代中心，然后重新把第一轮随机选出的聚类中心移到这一类的中心处
        c = torch.stack([x[a==k].mean(0) for k in range(ncluster)])
        # re-assign any poorly positioned codebook elements
        nanix = torch.any(torch.isnan(c), dim=1)
        ndead = nanix.sum().item()
        print('done step %d/%d, re-initialized %d dead clusters' % (i+1, niter, ndead))
        c[nanix] = x[torch.randperm(N)[:ndead]] # re-init dead clusters
    return c
