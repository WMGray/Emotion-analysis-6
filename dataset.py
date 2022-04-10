"""
1. 将各感情各自组成一个文件
2. 删除多余的符号 eg: !!!-->!
3. 数据去重
"""
import os
import re
import json
import codecs
import pandas as pd
from config import Config

# 可按数据集一个一个进行分类处理，但需要打开太多次Null、Like文件，效率不高
# 故先打开Null、Like文件，再打开数据集分类处理(或许有更好的方法)

# 其他（Null), 喜好(Like)，悲伤(Sad)，厌恶(Disgust)，愤怒(Anger)，高兴（Happiness）

def regular(file_name):
    """
    句子规范化，主要是对原始语料的句子进行一些标点符号的统一处理
    """
    file = codecs.open(file_name, encoding=Config.encoding)
    sen = []
    for index, line in enumerate(file):
        line = re.sub(r'…{1,100}', '…', line)
        line = re.sub(r'\.{3,100}', '…', line)
        line = re.sub(r'···{2,100}', '…', line)
        line = re.sub(r'\.{1,100}', '。', line)
        line = re.sub(r'。{1,100}', '。', line)
        line = re.sub(r'？{1,100}', '？', line)
        line = re.sub(r'!{1,100}', '！', line)
        line = re.sub(r'！{1,100}', '！', line)
        line = re.sub(r'~{1,100}', '～', line)
        line = re.sub(r'～{1,100}', '～', line)
        sen.append(line)
    file.close()

    file = codecs.open(file_name, "w", encoding=Config.encoding)
    for line in sen:
        file.write(line)
    file.close()


def qa_process(pairs, result, emotion_type):
    """处理问答对[[sentence,emotion],[],...]"""

    for pair in pairs:
        sentence, emotion = pair[0], pair[1]  # 句子，情感
        if emotion != emotion_type:  # 情感不匹配
            continue

        # 处理句子
        sentence = sentence.replace(' ', '').strip()  # 去掉空格
        sentence = sentence.replace('"', '“').replace(',', '，').replace(':', '：').replace('?', '？')  # 替换英文符号

        symbol = '@'  # 分隔符
        if symbol in sentence:  # 删除评论
            continue
        sentence = re.sub(r'\[[\u4e00-\u9fa5a-zA-Z]{1,100}\]', '', sentence)  # 去除表情

        if len(sentence) > 0:
            result.write(sentence + '\n')  # 写入文件


def emo_train_process(data_path, result_path, emotion_type):
    """对emo_train进行处理"""
    data = codecs.open(data_path, encoding=Config.encoding, errors='replace')
    result = codecs.open(result_path, 'a', encoding=Config.encoding)

    pairs = json.load(data)  # 加载json文件

    # 处理问答对
    qa_process(pairs, result, emotion_type)

    data.close()
    result.close()


def ecg_train_process(data_path, result_path, emotion_type):
    """对ecg_train进行处理"""
    data = codecs.open(data_path, encoding=Config.encoding, errors='replace')
    result = codecs.open(result_path, 'a', encoding=Config.encoding)

    pairs = json.load(data)  # 加载json文件

    for pair in pairs:  # 取出问答对
        qa_process(pair, result, emotion_type)

    data.close()
    result.close()


def ecg_test_process(data_path, result_path, emotion_type):
    """对ecg_test进行处理"""
    excel_data = pd.read_excel(data_path)
    result = codecs.open(result_path, 'a', encoding=Config.encoding)

    # 其他（Null)，喜好(Like)，悲伤(Sad)，厌恶(Disgust)，愤怒(Anger)，高兴（Happiness
    emo_dict = {'null': 0, 'like': 1, 'sad': 2, 'disgust': 3, 'angry': 4, 'happy': 5}
    sentence, emotion = list(excel_data['response']), list(excel_data['emotion'])

    pairs = []
    for sentence, emotion in zip(sentence, emotion):
        if type(sentence) != str:
            continue
        qe = [sentence, emo_dict[emotion]]
        pairs.append(qe)
    qa_process(pairs, result, emotion_type)
    result.close()


def file_remove_same(data_path, emotion_type):
    """数据去重"""
    file = codecs.open(data_path, encoding=Config.encoding, errors='replace')
    result_path = os.path.join(Config.classfication_root_path + '/' + emotion_type + '_new.tsv')
    result = codecs.open(result_path, 'a', encoding=Config.encoding)

    data = [item.strip() for item in file.readlines()]  # 针对最后一行没有换行符，与其他它行重复的情况
    new_data = list(set(data))
    result.writelines([item + '\n' for item in new_data if item])  # 针对去除文件中有多行空行的情况

    file.close()
    result.close()

def null(data_path, result_path):
    """写入Null情绪句子"""
    print("处理Null情绪")

    # 处理train.json
    print("处理train.json")
    emo_train_process(data_path[0], result_path, 0)

    # 处理ecg_train_data.json
    print("处理ecg_train_data.json")
    ecg_train_process(data_path[1], result_path, 0)

    # 处理ecg_test_data.xlsx
    print("处理ecg_test_data.xlsx")
    ecg_test_process(data_path[2], result_path, 0)

    # 去除多余符号
    print("去除多余符号")
    regular(result_path)

    # 数据去重
    print("数据去重")
    file_remove_same(result_path, 'Null')


def like(data_path, result_path):
    """写入Like情绪句子"""
    print("处理Like情绪")

    # 处理train.json
    print("处理train.json")
    emo_train_process(data_path[0], result_path, 1)

    # 处理ecg_train_data.json
    print("处理ecg_train_data.json")
    ecg_train_process(data_path[1], result_path, 1)

    # 处理ecg_test_data.xlsx
    print("处理ecg_test_data.xlsx")
    ecg_test_process(data_path[2], result_path, 1)

    # 去除多余符号
    print("去除多余符号")
    regular(result_path)

    # 数据去重
    print("数据去重")
    file_remove_same(result_path, 'Like')


def sad(data_path, result_path):
    """写入Sad情绪句子"""
    print("处理Sad情绪")

    # 处理train.json
    print("处理train.json")
    emo_train_process(data_path[0], result_path, 2)

    # 处理ecg_train_data.json
    print("处理ecg_train_data.json")
    ecg_train_process(data_path[1], result_path, 2)

    # 处理ecg_test_data.xlsx
    print("处理ecg_test_data.xlsx")
    ecg_test_process(data_path[2], result_path, 2)

    # 去除多余符号
    print("去除多余符号")
    regular(result_path)

    # 数据去重
    print("数据去重")
    file_remove_same(result_path, 'Sad')


def disgust(data_path, result_path):
    """写入Disgust情绪句子"""
    print("处理Disgust情绪")

    # 处理train.json
    print("处理train.json")
    emo_train_process(data_path[0], result_path, 3)

    # 处理ecg_train_data.json
    print("处理ecg_train_data.json")
    ecg_train_process(data_path[1], result_path, 3)

    # 处理ecg_test_data.xlsx
    print("处理ecg_test_data.xlsx")
    ecg_test_process(data_path[2], result_path, 3)

    # 去除多余符号
    print("去除多余符号")
    regular(result_path)

    # 数据去重
    print("数据去重")
    file_remove_same(result_path, 'Disgust')


def anger(data_path, result_path):
    """写入Anger情绪句子"""
    print("处理Anger情绪")

    # 处理train.json
    print("处理train.json")
    emo_train_process(data_path[0], result_path, 4)

    # 处理ecg_train_data.json
    print("处理ecg_train_data.json")
    ecg_train_process(data_path[1], result_path, 4)

    # 处理ecg_test_data.xlsx
    print("处理ecg_test_data.xlsx")
    ecg_test_process(data_path[2], result_path, 4)

    # 去除多余符号
    print("去除多余符号")
    regular(result_path)

    # 数据去重
    print("数据去重")
    file_remove_same(result_path, 'Anger')


def happiness(data_path, result_path):
    """写入Happiness情绪句子"""
    print("处理Happiness情绪")

    # 处理train.json
    print("处理train.json")
    emo_train_process(data_path[0], result_path, 5)

    # 处理ecg_train_data.json
    print("处理ecg_train_data.json")
    ecg_train_process(data_path[1], result_path, 5)

    # 处理ecg_test_data.xlsx
    print("处理ecg_test_data.xlsx")
    ecg_test_process(data_path[2], result_path, 5)

    # 去除多余符号
    print("去除多余符号")
    regular(result_path)

    # 数据去重
    print("数据去重")
    file_remove_same(result_path, 'Happiness')


def classification():
    """将不同情感分成六个文件"""
    data_path = Config.data_List
    result_path = Config.Emotion_List

    # 分类处理、去重
    null(data_path, result_path[0])
    like(data_path, result_path[1])
    sad(data_path, result_path[2])
    disgust(data_path, result_path[3])
    anger(data_path, result_path[4])
    happiness(data_path, result_path[5])




if __name__ == '__main__':
    classification()
