# 利用LinearRegression实现线性回归
import numpy as np
import sklearn.linear_model as lm  # 线性模型# 线性模型
import sklearn.metrics as sm  # 模型性能评价模块
import matplotlib.pyplot as mp

train_x = np.array([[230.1],
                    [44.5],
                    [17.2],
                    [151.5],
                    [180.8],
                    [57.5],
                    [120.2],
                    [8.6]])  # 输入集
train_y = np.array([22.1, 10.4, 9.3, 18.5, 12.9, 11.8, 13.2, 4.8])  # 输出集

# 创建线性回归器
model = lm.LinearRegression()
# 用已知输入、输出数据集训练回归器
model.fit(train_x, train_y)
# 根据训练模型预测输出
pred_y = model.predict(train_x)

print("coef_:", model.coef_)  # 系数
print("intercept_:", model.intercept_)  # 截距

# 可视化回归曲线
mp.figure('Linear Regression', facecolor='lightgray')
mp.title('Linear Regression', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')

# 绘制样本点
mp.scatter(train_x, train_y, c='blue', alpha=0.8, s=60, label='Sample')

# 绘制拟合直线
mp.plot(train_x,  # x坐标数据
        pred_y,  # y坐标数据
        c='orangered', label='Regression')

mp.legend()
mp.show()

print("电视广告投入300，销售收入为：", 300*model.coef_+model.intercept_)
print("电视广告投入400，销售收入为：", 400*model.coef_+model.intercept_)
print("电视广告投入500，销售收入为：", 500*model.coef_+model.intercept_)