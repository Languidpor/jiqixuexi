import numpy as np
import matplotlib.pyplot as mp
import sklearn.cluster as sc
#中心不明显的聚类
#规定生成的样本的数量
num=500
#rand()根据指定的参数的个数，生成不同维度的数组，500行1列
t = 2.5*np.pi*(1+2*np.random.rand(num,1))
#阿基米德螺线
x = 0.05*t*np.cos(t)
y = 0.05*t*np.sin(t)

n = 0.05*np.random.rand(num,2)
#x:500行2列的一个数组
x = np.hstack((x,y))+n
#创建凝聚层次聚类模型
model = sc.AgglomerativeClustering(n_clusters=4,linkage="complete")
model.fit(x)
mp.figure("",facecolor="snow")
mp.title("",fontsize=20)
mp.xlabel("x",fontsize=15)
mp.xlabel("y",fontsize=15)
mp.grid(linestyle=":")
mp.scatter(x[:,0],x[:,1],c=model.labels_,cmap="brg",s=80,marker="o",alpha=0.8)
mp.show()
