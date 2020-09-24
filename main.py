import sys
from click._compat import raw_input #接收行输入
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
import jieba

print("请输入源论文地址：")
f=open(raw_input(),"r",encoding="UTF-8")
f1=f.read()
print("请输入抄袭版论文地址：")
f=open(raw_input(),"r",encoding="UTF-8")
f2=f.read()
f.close()

def filter_words(file):
    pt = re.compile(u"[a-zA-Z0-0\u4e00-\u9fa5]")
    str = " "
    str = pt.sub("",str)
    str = jieba.cut(file,cut_all=False)
    return ' '.join(list(str)) #转化为TF矩阵

str1 = filter_words(f1)
str2 = filter_words(f2)

def jaccard_similarity(str1, str2):
    cv=CountVectorizer(tokenizer=lambda s: s.split())
    corpus=[str1, str2]
    vectors=cv.fit_transform(corpus).toarray()# 求交集
    numerator=np.sum(np.min(vectors, axis=0))# 求并集
    denominator=np.sum(np.max(vectors, axis=0))# 计算杰卡德系数
    return 1.0 * numerator / denominator

print("请输入输出答案地址：")
f=open(raw_input(),"w",encoding="UTF-8")
f.write( str(jaccard_similarity(str1, str2))[0:4] ) #输出查重结果
f.close()
