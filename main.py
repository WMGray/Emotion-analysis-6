import pandas as pd
import numpy as np
import jieba
import json
import codecs
import os
import yaml
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout,Activation
from tensorflow.keras.models import model_from_yaml
import multiprocessing
import matplotlib.pyplot as plt
from config import Config

vocab_dim = 200  # 词向量维度
maxlen = 100     # 序列最大长度
n_iterations = 10  # 迭代次数
n_exposures = 10    # 词频截断值
window_size = 7  # 窗口大小
batch_size = 128  # 批次大小
n_epoch = 4      # 迭代次数
input_length = 100  # 输入序列长度
cpu_count = multiprocessing.cpu_count()  #


def load_file():
    """加载数据"""
    # 加载数据集
    null = pd.read_csv(Config.new_List[0], header=None, index_col=None)  # null
    like = pd.read_csv(Config.new_List[1], header=None, index_col=None)  # like
    sad = pd.read_csv(Config.new_List[2], header=None, index_col=None)  # sad
    disgust = pd.read_csv(Config.new_List[3], header=None, index_col=None)  # disgust
    anger = pd.read_csv(Config.new_List[4], header=None, index_col=None)  # anger
    happy = pd.read_csv(Config.new_List[5], header=None, index_col=None)  # happy

    # concatenate 能够一次完成多个数组的拼接
    combined = np.concatenate((null[0], like[0], sad[0], disgust[0], anger[0], happy[0]))
    print(type(combined))
    # null-0 like-1 sad-2 disgust-3 anger-4 happy-5
    y = np.concatenate((np.zeros(len(null), dtype=int), np.ones(len(like), dtype=int),
                        np.ones(len(sad), dtype=int) * 2, np.ones(len(disgust), dtype=int) * 3,
                        np.ones(len(anger), dtype=int) * 4, np.ones(len(happy), dtype=int) * 5))

    return combined, y


def tokenizer(text):
    """对句子经行分词，并去掉换行符，结果保存再word.json文件中"""
    print("开始分词")
    word_json = codecs.open(Config.word_path, 'w', encoding=Config.encoding)

    text = [jieba.lcut(document.replace('\n', '')) for document in text]   # jieba.lcut() 生成一个列表

    json.dump(text, word_json)

    word_json.close()
    return text


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

        combined = parse_dataset(combined)   # 将combined中的数据转换为索引表示
        combined = sequence.pad_sequences(combined, maxlen=maxlen)  # 每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec, combined
    else:
        print('No data provided...')


def word2vec_train(combined):
    """创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引"""
    # size：是指特征向量的维度  min_count:对字典做截断. 词频少于min_count次数的单词会被丢弃掉
    # window：表示当前词与预测词在一个句子中的最大距离是多少  workers：参数控制训练的并行数。
    # iter： 迭代次数，默认为5
    model = Word2Vec(vector_size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     epochs=n_iterations)

    model.build_vocab(combined)     # 准备模型向量
    model.train(combined,total_examples=model.corpus_count,epochs=model.epochs)   # 训练词向量
    model.save('Data/Word2vec_model.pkl')  # 保存
    index_dict, word_vectors, combined = create_dictionaries(model=model, combined=combined)
    return index_dict, word_vectors, combined


def get_data(index_dict, word_vectors, combined, y):
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))    # 索引为0的词语，词向量全为0
    for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)

    print(x_train.shape,y_train.shape)

    return n_symbols,embedding_weights,x_train,y_train,x_test,y_test


##定义网络结构
def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
    print('Defining a Simple tensorflow.keras Model...')
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))  # Adding Input Length
    model.add(LSTM(units=50, activation='tanh'))
    model.add(Activation('hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense())
    model.add(Dense(6, activation='softmax')) # Dense=>全连接层,输出维度=6
    model.add(Activation('softmax'))

    print('Compiling the Model...')
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()

    print("Train...") # batch_size=128
    history = model.fit(x_train, y_train,
                        batch_size=batch_size, epochs=n_epoch,
                        verbose=1, validation_data=(x_test, y_test))
    acc = history.history['acc']
    loss = history.history['loss']
    epochs = range(1, len(acc) + 1)

    plt.title('Accuracy and Loss')
    plt.plot(epochs, acc, 'red', label='Training acc')
    plt.plot(epochs, loss, 'blue', label='Validation loss')
    plt.legend()
    plt.show()

    print("Evaluate...")
    score = model.evaluate(x_test, y_test, batch_size=batch_size)

    yaml_string = model.to_yaml()
    with open('model/lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True) )
    model.save_weights('model/lstm.h5')
    print('Test score:', score)

#训练模型，并保存
print('Loading Data...')
combined,y =load_file()
print(len(combined), len(y))

print('Tokenising...')
if os.path.exists(Config.word_path):
    print("文件已存在")
    word_json = codecs.open(Config.word_path, encoding=Config.encoding, errors='replace')
    combined = json.load(word_json)
    word_json.close()
else:
    combined = tokenizer(combined)
print('Training a Word2vec model...')
index_dict, word_vectors, combined = word2vec_train(combined)

print('Setting up Arrays for tensorflow.keras Embedding Layer...')
n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, combined, y)

print("x_train.shape and y_train.shape:")
print(x_train.shape, y_train.shape)
train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)


