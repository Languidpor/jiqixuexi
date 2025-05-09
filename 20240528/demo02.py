import nltk.tokenize as tk #自然语言处理
import sklearn.feature_extraction.text as ft #词袋模型，TF-IDF模型
import sklearn.naive_bayes as sn#朴素贝叶斯
import numpy as np
# 1 按照语句对段落进行拆分， 参数是doc段落
doc = "This hotel is very bad very bad. " \
      "The toilet in this hotel smells bad. " \
      "The environment of this hotel is very good."
sent_list = tk.sent_tokenize(doc)
# 2 构建一个词袋模型训练器
cv = ft.CountVectorizer()
# 3 训练这个训练器(fit_transfrom), 参数是语句列表，得到词袋模型
bow = cv.fit_transform(sent_list)
# 4 获取词频-逆文档频率训练器TF-IDF
tt = ft.TfidfTransformer()
# 5 训练这个训练器(fit_transfrom),参数就是词袋模型—此处就得到输入x
x = tt.fit_transform(bow)
# 6 对于x输入样本打标签，作为y
y = np.array([0,0,1],dtype=int)
# 7 构建机器学习分类模型并训练，多项式朴素贝叶斯(适用于绝大多数离散数据的场景),
model  = sn.MultinomialNB()
model.fit(x,y)
# 8 准备测试段落
doc1 = "environment of good "
# 9 将测试段落进行语句拆分
test_sent = tk.sent_tokenize(doc1)
# 10 用语句训练词袋模型(transform)，得到词袋模型
bow = cv.transform(test_sent)
# 11 用词袋模型训练词频-逆文档频率模型(transform)，得到test_x
test_x = tt.transform(bow)
# 12 用模型预测test_x
pred_y = model.predict(test_x)
print(pred_y)

