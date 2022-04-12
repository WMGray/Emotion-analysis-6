# 预测
import jieba
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence

import yaml
from keras.models import model_from_yaml
np.random.seed(1337)  # For Reproducibility
import sys
sys.setrecursionlimit(1000000)

# define parameters
maxlen = 100

def create_dictionaries(model=None, combined=None):
    """创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引"""
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()  # 创建一个空的词典,构建 word<->id 映射
        gensim_dict.doc2bow(list(model.wv.index_to_key),
                                allow_update=True)  # 构建词袋，每个单词对应一个id，词袋中的单词不重复
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 所有频数超过10的词语的索引字典
        w2vec = {word: model.wv[word] for word in w2indx.keys()}  # 所有频数超过10的词语的词向量字典
        def parse_dataset(combined):
            """将combined中的数据转换为索引表示"""
            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data

        combined = parse_dataset(combined)  # 将combined中的数据转换为索引表示
        combined = sequence.pad_sequences(combined, maxlen=maxlen)  # 每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec, combined
    else:
        print('No data provided...')


def input_transform(string):
    words=jieba.lcut(string)
    words=np.array(words).reshape(1,-1)
    model=Word2Vec.load('Data/Word2vec_model.pkl')
    _,_,combined=create_dictionaries(model,words)
    return combined


def lstm_predict(string):
    print ('loading model......')
    with open('model/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f, Loader=yaml.FullLoader)
    model = model_from_yaml(yaml_string)
    
    print ('loading weights......')
    model.load_weights('model/lstm.h5')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    data=input_transform(string)
    print(data)
    data.reshape(1,-1)
    print(data)
    result=np.argmax(model.predict(data), axis=-1)
    print (result )# [[1]]
    if result[0] == 0:
      print(string, '其他')
    elif result[0] ==1:
      print(string, '喜欢')
    elif result[0] ==2:
      print(string, '悲伤')
    elif result[0] ==3:
      print(string, '厌恶')
    elif result[0] ==4:
      print(string, '愤怒')
    elif result[0] ==5:
      print(string, '开心')


if __name__ == '__main__':
  lstm_predict('我喜欢你')