'''文本向量化处理---->词袋模型'''
import nltk.tokenize as tk
doc = "This hotel is very bad very bad. " \
      "The toilet in this hotel smells bad. " \
      "The environment of this hotel is very good."
#按照语句进行拆分
sent_list = tk.sent_tokenize(doc)
print(sent_list)

import sklearn.feature_extraction.text as ft
#构建词袋模型训练器
cv = ft.CountVectorizer()
#训练这个训练器
bow = cv.fit_transform(sent_list).toarray()
print(bow)
#副产品，可以得到特证名
words = cv.get_feature_names()
print(words)
#词袋模型举证的归一化处理
import sklearn.preprocessing as sp
tf = sp.normalize(bow,norm="l1")
print(tf)








