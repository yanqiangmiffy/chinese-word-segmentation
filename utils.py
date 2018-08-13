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


    def process_data(self,filename):
        """
        预处理训练数据
        :param filename:
        :return:
        """
        texts=self.read_data(filename) # 从训练文件读取所有行
        texts="".join(map(self.clean,texts)) # clean
        sentences=re.split('[，。！？、‘’“”]/[bems]',texts) # 切分句子

        lines=list()
        labels=list()
        for sentence in sentences:
            result=1



    def read_data(self,filename):
        """
        读取数据
        :param filename:
        :return: 返回句子列表
        """
        return [line.strip() for line in codecs.open(filename,'r','utf-8').readlines()] # strip()

    def clean(self,text):
        """
        清洗文本 去掉不必要的引号
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

if __name__ == '__main__':
    data=DataHelper('data')
    lines=data.read_data('data/msr_train.txt')
    print(lines)