#练习：利用支持向量回归预测体育场馆周边交通流量。
# 样本特征分别为：星期、时间、对手球队、
# 棒球比赛是否正在进行、通行汽车数量。
# traffic.txt
import numpy as np
#数据的拆分，分为训练数据和测试数据
import sklearn.model_selection as ms
import sklearn.svm as svm
#模型的评估
import sklearn.metrics as sm
#数据的预处理
import sklearn.preprocessing as sp
#自定义一个编码器，将str-->int, int-->str
class MyEncoder():
    def fit_transform(self,x):
        return x.astype(int)
    def inverse_transfrom(self,x):
        return x.astype(str)
#读样本文件，预处理数据(编码，每一行对应5个编码器，而且要把编码器存储起来)
data=[]
with open("traffic.txt","r") as f:
    for line in f.readlines():
        data.append(line.strip().split(","))
#转置，变成了5行17000+列的数组
data = np.array(data).T
#创建模型对象，并训练模型
encoders = []#存储编码器
x = []#存储输入，也就是4个属性
#y #存储输出，也就是最后一行，标签
for row in range(len(data)):
    if row < len(data)-1:#前四行
        encoder = sp.LabelEncoder()
        x.append(encoder.fit_transform(data[row]))
        encoders.append(encoder)
    else:#最后一行
        myEncoder = MyEncoder()
        y = myEncoder.fit_transform(data[row])
        encoders.append(myEncoder)
x = np.array(x).T
#model_selection 划分样本
train_x,test_x,train_y,test_y = \
    ms.train_test_split(x,y,test_size=0.25,random_state=5)
#C正则化系数，epsilon对于错误分类的惩罚力度
model = svm.SVR(kernel="rbf", C=10, epsilon=0.2)
model.fit(train_x,train_y)
#评估模型
pred_y = model.predict(test_x)#预测值
r2 = sm.r2_score(test_y,pred_y)
print(r2)
#造一个数据，然后进行编码，然后进行预测
