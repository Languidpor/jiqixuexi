'''文本分句(划分样本)、文本分词(统计词频、逆文档频率)'''
#自然语言处理工具包
import nltk.tokenize as tk
doc = "Are you curious about tokenization? " \
      "Let's see how it works! " \
      "we need to analyze a couple of " \
      "sentences with punctuations to see it in action."
print(doc)
#按照语句进行拆分(使用的什么标准进行从拆分，自己测试)，返回的语句列表
sent_list = tk.sent_tokenize(doc)
print(sent_list)
#按照单词进行拆分(Let's会不会拆分成两个单词)，返回的是单词列表,包含标点符号
word_list = tk.word_tokenize(doc)
print(word_list)
#单词标点分割
tokenizer = tk.WordPunctTokenizer()
word_list = tokenizer.tokenize(doc)
print(word_list)








