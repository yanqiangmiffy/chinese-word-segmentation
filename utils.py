import codecs
import numpy as np
import re
from tqdm import tqdm
import pandas as pd
from itertools import chain
import os
import pickle


class DataHelper():
    def __init__(self,model_path):
        self.max_len=32
        self.vocab_size=None
        self.df_data=None
        self.word2id=None # 词转id
        self.id2word=None # id转词
        self.tag2id=None
        self.id2tag=None
        self.X=None
        self.y=None
        self.model_path=model_path # 存储路径

    def read_data(self, filename):
        """
        读取数据  （process_data函数中将会用到）
        :param filename:
        :return: 返回句子列表
        """
        return [line.strip() for line in codecs.open(filename, 'r', 'utf-8').readlines()]  # strip()

    def clean(self, text):
        """
        清洗文本 去掉不必要的引号 （process_data函数中将会用到）
        :param text:
        :return:
        """
        if '“' not in text:  # 句子中间的引号不应去掉
            return text.replace(' ”', '')
        elif '”' not in text:
            return text.replace('“ ', '')
        elif '‘' not in text:
            return text.replace(' ’', '')
        elif '’' not in text:
            return text.replace('‘ ', '')
        else:
            return text

    def get_xy(self,sentence):
        """
        将 sentence 处理成 [word1, word2, ..wordn], [tag1, tag2, ...tagn]
        :param sentence:
        :return:
        """
        words_tags=re.findall('(.)/(.)',sentence)
        if words_tags:
            words_tags=np.asarray(words_tags)
            words=words_tags[:,0]
            tags=words_tags[:,1]
            return words,tags

    def transform(self,df_data):
        """
        word、tags与id之间的转换
        :param df_data:
        :return:
        """
        # 1. 用chain(*lists)函数将多个list拼接起来
        all_words=list(chain(*df_data['words'].values))

        # 2. 统计所有 word
        sr_words=pd.Series(all_words)
        sr_words=sr_words.value_counts()
        set_words=sr_words.index
        self.vocab_size=len(set_words)
        set_ids=range(1,len(set_words)+1) # 0 为填充值
        tags=['x','s','b','m','e']
        tags_ids=range(len(tags))

        # 3. 构建 words 和 tags 都转为数值 id 的映射（使用 Series 比 dict 更加方便）
        word2id=pd.Series(set_ids,index=set_words)
        id2word=pd.Series(set_words,index=set_ids)
        tag2id=pd.Series(tags_ids,index=tags)
        id2tag=pd.Series(tags,index=tags_ids)
        self.word2id=word2id
        self.id2word=id2word
        self.tag2id=tag2id
        self.id2tag=id2tag

    def process_data(self,filename):
        """
        预处理训练数据
        :param filename:
        :return:
        """
        texts=self.read_data(filename) # 从训练文件读取所有行
        texts="".join(map(self.clean,texts)) # clean
        sentences=re.split('[，。！？、‘’“”]/[bems]',texts) # 切分句子
        datas =list()
        labels=list()

        for sentence in tqdm(sentences):
            result=self.get_xy(sentence)
            if result:
                datas.append(result[0])
                labels.append(result[1])

        df_data=pd.DataFrame({'words':datas,'labels':labels},index=range(len(datas)))
        # 句子长度
        df_data['sentence_len']=df_data['words'].apply(lambda words:len(words))







if __name__ == '__main__':
    data=DataHelper('data')
    data.process_data('data/msr_train.txt')
