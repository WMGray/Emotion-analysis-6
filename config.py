import os

class Config:
    """基本设置"""
    encoding = "utf-8"  # 编码

    # 文件路径
    root_path = "Data"  # 根目录
    classfication_root_path = "Data/classfication"  # 分类数据目录

    # 文件名
    emo_train_path = os.path.join(root_path, "train.json")  # 情感分类数据集

    ecg_train_path = os.path.join(root_path, "ecg_train_data.json")  # ecg训练集
    ecg_test_path = os.path.join(root_path, "ecg_test_data.xlsx")  # ecg测试集

    data_List = [emo_train_path, ecg_train_path, ecg_test_path]  # 数据集列表

    # 已分类数据
    # 其他（Null), 喜好(Like)，悲伤(Sad)，厌恶(Disgust)，愤怒(Anger)，高兴（Happiness）
    Null_path = os.path.join(classfication_root_path, "Null.tsv")  # Null
    Like_path = os.path.join(classfication_root_path, "Like.tsv")  # 喜好
    Sad_path = os.path.join(classfication_root_path, "Sad.tsv")  # 悲伤
    Disgust_path = os.path.join(classfication_root_path, "Disgust.tsv")  # 厌恶
    Anger_path = os.path.join(classfication_root_path, "Anger.tsv")  # 愤怒
    Happiness_path = os.path.join(classfication_root_path, "Happiness.tsv")  # 高兴
    Emotion_List = [Null_path, Like_path, Sad_path, Disgust_path, Anger_path, Happiness_path]  # 情感分类

    # 已清洗数据
    null_new_path = os.path.join(classfication_root_path, "Null_new.tsv")  # Null
    like_new_path = os.path.join(classfication_root_path, "Like_new.tsv")  # 喜好
    sad_new_path = os.path.join(classfication_root_path, "Sad_new.tsv")  # 悲伤
    disgust_new_path = os.path.join(classfication_root_path, "Disgust_new.tsv")  # 厌恶
    anger_new_path = os.path.join(classfication_root_path, "Anger_new.tsv")  # 愤怒
    happiness_new_path = os.path.join(classfication_root_path, "Happiness_new.tsv")  # 高兴
    new_List = [null_new_path, like_new_path, sad_new_path, disgust_new_path, anger_new_path, happiness_new_path]  # 已分类数据

    # 分词数据
    word_path = os.path.join(root_path, "word.json")  # 分词数据--列表
    w2indx_path = os.path.join(root_path, "w2indx.json")  # 分词索引--字典
    w2vec_path = os.path.join(root_path, "w2vec.json")  # 分词向量--列表
    combined_path = os.path.join(root_path, "combined.json")  # 分词数据--np
    # stopwords
    stop_words_path = os.path.join(root_path, "StopWord.txt")  # 停用词
