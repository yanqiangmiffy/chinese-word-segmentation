str='偶尔  有  老乡  拥  上来  想  看  “  大官  ”  ，  立即  会  遭到  “  闪开  ！'
# str='“  种菜  ，  也  有  烦恼  ，  那  是  累  的  时候  ；  另外  ，  大  棚  菜  在  降价  。'

# def clean(s):
#     if '“' not in s:  # 句子中间的引号不应去掉
#         return s.replace(' ”', '')
#     elif '”' not in s:
#         return s.replace('“ ', '')
#     elif '‘' not in s:
#         return s.replace(' ’', '')
#     elif '’' not in s:
#         return s.replace('‘ ', '')
#     else:
#         return s

# from itertools import chain
# s1=['i','l','o','v','e']
# s2=['y','o','u']
# s=[s1,s2]
#
# all_letters=chain(*s)
# print(all_letters,type(all_letters))
# print(list(all_letters))


words=["我","北京","天安门"]
word2id={word:index for index,word in enumerate(words)}
id2word={index:word for index,word in enumerate(words)}
print(word2id)
print(id2word)

import pandas as pd
ids=range(len(words))
word2id=pd.Series(ids,index=words).to_dict()
id2word=pd.Series(words,index=ids).to_dict()
print(word2id)
print(id2word)