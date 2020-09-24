
import sys

from click._compat import raw_input
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import jieba
f=open(raw_input(),"r",encoding="UTF-8")
s1=f.read()#输入s1
f=open(raw_input(),"r",encoding="UTF-8")
s2=f.read()#输入s2
f.close()
sign='· ~ ！ @ # ￥ % …… & * （ ） —— - + = ] } [ { \ | ; :  " " < > , . /  ？ $ ^ _'
for word in sign:
    s1=s1.replace(word,'') #循环文本，删去标点符号
def jaccard_similarity(s1, s2):#jaccard
    def add_space(s):
        s_list=jieba.cut(s,cut_all=False)
        return ' '.join(list(s_list))# 将字中间加入空格
    s1, s2=add_space(s1), add_space(s2)# 转化为TF矩阵
    cv=CountVectorizer(tokenizer=lambda s: s.split())
    corpus=[s1, s2]
    vectors=cv.fit_transform(corpus).toarray()# 求交集
    numerator=np.sum(np.min(vectors, axis=0))# 求并集
    denominator=np.sum(np.max(vectors, axis=0))# 计算杰卡德系数
    return 1.0 * numerator / denominator

#print("orig_0.8_dis_1的测试结果：{:f}".format(jaccard_similarity(s1, s2)))#输出查重结果
f=open(raw_input(),"w",encoding="UTF-8")
f.write( str(jaccard_similarity(s1, s2))[0:4] )
f.close()
