import os

class Config:
    """基本设置"""
    encoding = "utf-8"  # 编码

    # 文件路径
    root_path = "source_data"  # 源数据目录
    classfication_root_path = "target_data"  # 已分类数据目录
    word2vec_path = "word2vec"  # 存放word2vec模型

    # 文件名
    emo_train_path = os.path.join(root_path, "train.json")  # 情感分类数据集

    ecg_train_path = os.path.join(root_path, "ECG_dataset/ecg_train_data.json")  # ecg训练集
    ecg_test_path = os.path.join(root_path, "ECG_dataset/ecg_test_data.xlsx")  # ecg测试集

    data_List = [emo_train_path, ecg_train_path, ecg_test_path]  # 数据集列表

    # 已分类数据
    # 其他（Null), 喜好(Like)，悲伤(Sad)，厌恶(Disgust)，愤怒(Anger)，高兴（Happiness）
    Null_path = os.path.join(classfication_root_path, "null.tsv")  # Null
    Like_path = os.path.join(classfication_root_path, "like.tsv")  # 喜好
    Sad_path = os.path.join(classfication_root_path, "sad.tsv")  # 悲伤
    Disgust_path = os.path.join(classfication_root_path, "disgust.tsv")  # 厌恶
    Anger_path = os.path.join(classfication_root_path, "anger.tsv")  # 愤怒
    Happiness_path = os.path.join(classfication_root_path, "happiness.tsv")  # 高兴
    Emotion_List = [Null_path, Like_path, Sad_path, Disgust_path, Anger_path, Happiness_path]  # 情感分类


    # 分词数据
    w2indx_path = os.path.join(word2vec_path, "w2indx.json")  # 分词索引--字典
    w2vec_path = os.path.join(word2vec_path, "w2vec.json")  # 分词向量--列表
    combined_path = os.path.join(word2vec_path, "combined.json")  # 句子 词id eg:[0,256,25,23,1]

    word2vec_path = os.path.join(word2vec_path, "word2vec.model")

