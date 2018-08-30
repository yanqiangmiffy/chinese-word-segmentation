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

    def X_padding(self,words):
        """
        把words转为id形式，并自动补全位 max_len长度
        :param words:
        :return:
        """
        max_len=self.max_len
        word2id=self.word2id
        ids=list(word2id[words])
        if len(ids)>=max_len: # 长则去掉
            return ids[:max_len]
        ids.extend([0]*(max_len-len(ids))) # 短则补全
        return ids

    def y_padding(self,tags):
        """
        把tags转为id形式，并自动补全位max_len长度
        :param tags:
        :return:
        """
        max_len=self.max_len
        tag2id=self.tag2id
        ids=list(tag2id[tags])
        if len(ids)>max_len: # 长则去掉
            return ids[:max_len]
        ids.extend([0]*(max_len-len(ids))) # 短则补全
        return ids

    def save_model(self,model_path):
        """
        保存处理好的数据
        :param model_path:
        :return:
        """
        with open(model_path,'wb') as out_data:
            pickle.dump(self.X,out_data,pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.y,out_data,pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.word2id,out_data,pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.id2word,out_data,pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.tag2id,out_data,pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.id2tag,out_data,pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.labels,out_data,pickle.HIGHEST_PROTOCOL)
            print("**Finished saving the data.")

    def load_model(self,model_path):
        """
        如果存在保存的数据，就直接加载
        :param model_path:
        :return:
        """
        with open(model_path,'rb') as in_data:
            self.X=pickle.load(in_data)
            self.y=pickle.load(in_data)
            self.word2id=pickle.load(in_data)
            self.id2word=pickle.load(in_data)
            self.tag2id=pickle.load(in_data)
            self.id2tag=pickle.load(in_data)
            self.labels=pickle.load(in_data)


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
        self.transform(df_data)
        df_data['X']=df_data['words'].apply(self.X_padding)
        df_data['y']=df_data['labels'].apply(self.y_padding)
        self.df_data=df_data
        X=np.asarray(list(df_data['X'].values))
        y=np.asarray(list(df_data['y'].values))
        self.X=X
        self.y=y
        self.labels=labels
        self.save_model(self.model_path)


    def text2ids(self,text):
        """
        将文本转为id表示
        :param text:
        :return:
        """
        words=list(text)
        max_len=self.max_len
        ids=list(self.word2id[words])

        if len(ids) >= max_len:  # 长则弃掉
            print ("输出片段超过%d部分无法处理" % (max_len))
            return ids[:max_len]
        ids.extend([0]*(max_len-len(ids))) # 短则补全
        ids = np.asarray(ids).reshape([-1, max_len])
        return ids

class BatchGenerator(object):

    """ Construct a Data generator. The input X, y should be ndarray or list like type.

    Example:
        Data_train = BatchGenerator(X=X_train_all, y=y_train_all, shuffle=False)
        Data_test = BatchGenerator(X=X_test_all, y=y_test_all, shuffle=False)
        X = Data_train.X
        y = Data_train.y
        or:
        X_batch, y_batch = Data_train.next_batch(batch_size)
     """

    def __init__(self,X,y,shuffle=False):
        if type(X)!=np.ndarray:
            X=np.asarray(X)
        if type(y)!=np.ndarray:
            y=np.asarray(y)

        self._X=X
        self._y=y
        self._epochs_completed=0
        self._index_in_epoch=0
        self._number_examples=self._X.shape[0]
        self._shuffle=shuffle
        if self._shuffle:
            new_index=np.random.permutation(self._number_examples)
            self._X=self._X[new_index]
            self._y=self._y[new_index]

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def num_examples(self):
        return self._number_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self,batch_size):
        """ Return the next 'batch_size' examples from this data set."""
        start=self._index_in_epoch
        self._index_in_epoch=+batch_size

        if self._index_in_epoch > self._number_examples:
            # finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if self._shuffle:
                new_index = np.random.permutation(self._number_examples)
                self._X = self._X[new_index]
                self._y = self._y[new_index]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._number_examples
        end = self._index_in_epoch
        return self._X[start:end], self._y[start:end]
# if __name__ == '__main__':
#     data=DataHelper('model/vocab_model.pkl')
#     data.process_data('data/msr_train.txt')
