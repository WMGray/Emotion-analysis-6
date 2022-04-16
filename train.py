import pandas as pd
import numpy as np
import jieba
import json
import codecs
import os
import yaml
import tensorflow as tf
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.callbacks import ReduceLROnPlateau
import multiprocessing
import matplotlib.pyplot as plt
from config import Config

vocab_dim = 200  # 词向量维度
maxlen = 100  # 序列最大长度
n_iterations = 10  # 迭代次数
n_exposures = 10  # 词频截断值
window_size = 7  # 窗口大小
batch_size = 1024  # 批次大小
n_epoch = 25  # 迭代次数
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


def tokenizer(combined):
    """对句子经行分词，并去掉换行符，结果保存再word.json文件中"""
    print("开始分词")
    word_json = codecs.open(Config.word_path, 'w', encoding=Config.encoding)
    stopwords = codecs.open(Config.stop_words_path, encoding='utf-8', errors='replace')  # 加载停用词
    stop_words, text = [], []
    for line in stopwords:
        stop_words.append(line.strip())

    for document in combined:
        word_list = jieba.lcut(document.replace('\n', ''))
        for w in word_list:
            if w in stop_words:
                word_list.remove(w)
        text.append(word_list)

    json.dump(text, word_json)

    word_json.close()
    return text


def create_dictionaries(model=None, combined=None):
    """创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引"""
    if (combined is not None) and (model is not None):
        if os.path.exists(Config.combined_path) and os.path.exists(Config.w2vec_path) and os.path.exists(
                Config.w2indx_path):  # 已写入
            print("加载词典")
            w2indx_json = codecs.open(Config.w2indx_path, 'r', encoding=Config.encoding)
            w2vec_json = codecs.open(Config.w2vec_path, 'r', encoding=Config.encoding)
            combined_json = codecs.open(Config.combined_path, 'r', encoding=Config.encoding)

            w2indx = json.load(w2indx_json)
            W2VEC = json.load(w2vec_json)  # 加载的数组为list，需转换为numppy数组
            combined = np.loadtxt(combined_json)

            # 转换
            w2vec = dict()
            for key, value in W2VEC.items():
                w2vec[key] = np.asarray(value)

            w2indx_json.close()
            w2vec_json.close()
            combined_json.close()

            return w2indx, w2vec, combined
        else:
            gensim_dict = Dictionary()  # 创建一个空的词典,构建 word<->id 映射
            gensim_dict.doc2bow(list(model.wv.index_to_key),
                                allow_update=True)  # 构建词袋，每个单词对应一个id，词袋中的单词不重复
            w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 所有频数超过10的词语的索引字典
            w2vec = {word: model.wv[word] for word in w2indx.keys()}  # 所有频数超过10的词语的词向量字典
            W2VEC = {word: model.wv[word].tolist() for word in w2indx.keys()}  # 将numpy数组转换为list存储

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

            w2indx_json = codecs.open(Config.w2indx_path, 'w', encoding=Config.encoding)
            w2vec_json = codecs.open(Config.w2vec_path, 'w', encoding=Config.encoding)
            combined_json = codecs.open(Config.combined_path, 'w', encoding=Config.encoding)

            json.dump(w2indx, w2indx_json)
            json.dump(W2VEC, w2vec_json)
            np.savetxt(combined_json, combined)  # numpy.ndarrayi

            w2indx_json.close()
            w2vec_json.close()
            combined_json.close()

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

    model.build_vocab(combined)  # 准备模型向量
    model.train(combined, total_examples=model.corpus_count, epochs=model.epochs)  # 训练词向量
    model.save('Data/Word2vec_model.pkl')  # 保存
    index_dict, word_vectors, combined = create_dictionaries(model=model, combined=combined)
    return index_dict, word_vectors, combined


def get_data(index_dict, word_vectors, combined, y):
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))  # 索引为0的词语，词向量全为0
    for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)

    print(x_train.shape, y_train.shape)

    return n_symbols, embedding_weights, x_train, y_train, x_test, y_test


# 绘制曲线
def plot_curve(history):
    # 绘制训练 & 验证的准确率值
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # 绘制训练 & 验证的损失值
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


##定义网络结构
def train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test):
    print('Defining a Simple tensorflow.keras Model...')
    # 定义神经网络模型
    model = Sequential()
    # 使用预训练的词向量 trainable=True 表示可训练
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,  # 
                        trainable=True,
                        weights=[embedding_weights],
                        input_length=input_length))  # Adding Input Length
    model.add(SpatialDropout1D(0.3))  # 功能与 Dropout 相同，但它会丢弃整个 1D 的特征图而不是丢弃单个元素。
    model.add(Conv1D(activation='tanh', padding='same', filters=32, kernel_size=7))  # 卷积层
    model.add(MaxPool1D(pool_size=2))  # 池化层
    model.add(Bidirectional(LSTM(100, return_sequences=True, activation='tanh'), merge_mode='concat'))  # 双向循环神经网络层
    model.add(Dropout(0.2))
    model.add(MaxPool1D(pool_size=2))
    model.add(LSTM(50, activation='tanh'))  # LSTM 层
    model.add(Flatten())  # 扁平层
    model.add(Dropout(0.2))
    model.add(BatchNormalization())  # 批标准化
    model.add(Dense(6, activation='tanh'))  # 加入偏置项
    # model.add(Dense(3, activation='tanh',            #加入偏置项
    #                 kernel_regularizer=regularizers.l2(0.01),
    #                 activity_regularizer=regularizers.l2(0.01)))
    # model.add(Activation('softmax'))

    print('Compiling the Model...')
    # loss-目标函数（categorical_crossentropy-多分类，binary_crossentropy-二分类）
    # optimizer-指定模型训练的优化器
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()

    print("Train...")  # batch_size=128
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto')
    history = model.fit(x_train, y_train,
                        batch_size=batch_size, epochs=n_epoch,
                        validation_split=0.01, callbacks=[reduce_lr],
                        verbose=1, validation_data=(x_test, y_test))

    # 绘制
    plot_curve(history)

    print("Evaluate...")
    score = model.evaluate(x_test, y_test, batch_size=batch_size)  # 在测试模式，返回误差值和评估标准值。

    yaml_string = model.to_yaml()
    with open('model/lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights('model/lstm.h5')
    print('Test score:', score)


def train():
    # 训练模型，并保存
    print('Loading Data...')
    combined, y = load_file()
    print(len(combined), len(y))

    print('Tokenising...')
    if os.path.exists(Config.word_path):
        print("文件已存在")
        word_json = codecs.open(Config.word_path, encoding=Config.encoding, errors='replace')
        combined = json.load(word_json)
        word_json.close()
    else:
        print(len(combined))
        combined = tokenizer(combined)
    print('Training a Word2vec model...')
    index_dict, word_vectors, combined = word2vec_train(combined)

    print('Setting up Arrays for tensorflow.keras Embedding Layer...')
    n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, combined, y)

    print("x_train.shape and y_train.shape:")
    print(x_train.shape, y_train.shape)
    train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    train()
