# 情感六分类

# 数据集
1. 数据集：[情感分类数据集](https://www.biendata.xyz/ccf_tcci2018/datasets/emotion/)
          [情感对话生成数据集](https://www.biendata.xyz/ccf_tcci2018/datasets/ecg/)
2. **情感分类数据集**：其他（Null), 喜好(Like)，悲伤(Sad)，厌恶(Disgust)，愤怒(Anger)，高兴（Happiness）六类，依次标号为0到5。
             NLPCC Emotion Classification Challenge（训练数据中17113条，测试数据中2242条）和微博数据筛选后人工标注(训练数据中23000条，测试数据中2500条)
   **情感对话生成数据集**:其他（Null)，喜好(Like)，悲伤(Sad)，厌恶(Disgust)，愤怒(Anger)，高兴（Happiness）六类
# 数据预处理
1. **情感分类数据集**：删除空格，替换英文符号，删除表情(eg: [流泪])，分类

   |  Emotion  | Size  |
   | :-------: | :---: |
   |   Null    | 13869 |
   |   Like    | 6496  |
   |    Sad    | 5285  |
   |  Disgust  | 5891  |
   |   Anger   | 3122  |
   | Happiness | 4767  |
   | **Total** | 39430 |

2. **情感对话生成数据集**：

   ECG-train：列表中每一个元素都是列表，为一个问答对。删除空格，替换英文符号，分类

   |  Emotion  |  Size   |
   | :-------: | :-----: |
   |   Null    | 537398  |
   |   Like    | 460849  |
   |    Sad    | 311269  |
   |  Disgust  | 386466  |
   |   Anger   | 220559  |
   | Happiness | 321873  |
   | **Total** | 2238414 |
   
   ECG-test

   |  Emotion  | Size |
   | :-------: | :--: |
   |   Null    |  0   |
   |   Like    | 1459 |
   |    Sad    | 1204 |
   |  Disgust  | 1412 |
   |   Anger   | 715  |
   | Happiness | 1433 |
   | **Total** | 6223 |
   
3. 数据清洗:删除重复的符号、去重

   |  Emotion  |       Siz       |      |
   | :-------: | :-------------: | :--: |
   |   Null    | 551267-->522995 | 0.24 |
   |   Like    | 468804-->440551 | 0.2  |
   |    Sad    | 317758-->301459 | 0.13 |
   |  Disgust  | 393769-->381264 | 0.17 |
   |   Anger   | 224396-->216072 | 0.09 |
   | Happiness | 328073-->302244 | 0.14 |
   | **Total** |     2164585     |      |

- 由于某些不可知原因，每次去重后数据位置发生变化，但数据量并不会发生改变(相较于未去重减少，只是去重数据位置不固定)
- 在所有数据基础上训练30轮，训练集准确率为70%左右
- 因所有数据训练时间过长，故每个随机选取20w进行试验

修改：

- https://cache.one/read/16905246
- https://www.kaggle.com/general/197993
- https://blog.csdn.net/weixin_42943494/article/details/108127523
- https://blog.csdn.net/cyz52/article/details/90454158
- https://blog.csdn.net/lcy6239/article/details/115786432
- https://blog.csdn.net/qsx123432/article/details/120583529
- ![image-20220409163304407](C:\Users\14667\AppData\Roaming\Typora\typora-user-images\image-20220409163304407.png)

参考：

- [基于BiLSTM的对话文本情感分析](http://www.chenjianqu.com/show-38.html)
- https://github.com/lidianxiang/predict_chinese_sentiment_in_tensorflow（可参考）
- https://github.com/luanshiqiguo/emotion-analysis-5
- https://github.com/DLLXW/MultiClassify_LSTM_ForChinese
