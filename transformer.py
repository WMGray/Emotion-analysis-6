#  transformer
import torch
import torch.nn as nn
from torch import optim
from torch.utils import data as Data
import numpy as np

source_vocab_size = 43655
target_vocab_size = 43655

# transformer 超参数
d_model = 512  # embedding size  词嵌入维度
max_len = 20  # max length of sentences    句子的最大长度
d_ff = 2048  # feedforward neural network dimension  前馈神经网络隐藏层大小
d_k = d_v = 64  # dimension of q、k、v     Q、K、V 维度
n_layers = 6  # number of encoder and decoder layers  编码器、解码器层数
n_heads = 8  # number of heads in multihead attention    注意力头数
p_drop = 0.1  # probability of dropout    Dropout的概率


# MASK:
# 1. 数据中使用了padding, 不希望pad被加入到注意力中进行计算的Pad Mask for Attention
# 2. 保证Decoder自回归信息不泄露的Subsequent Mask for Decoder
# 3. 在Encoder中使用Mask, 是为了将encoder_input中没有内容而打上PAD的部分进行Mask, 方便矩阵运算.
# 4. 在Decoder中使用Mask, 可能是在Decoder的自注意力对decoder_input的PAD进行Mask, 也有可能是对Encoder-Decoder自注意力时对
#    encoder_input和decoder_input的PAD进行Mask.

# Pad Mask for Attention
def get_attn_pad_mask(seq_q, seq_k):
    """
    Padding, because of unequal in source_len and target_len.

    parameters:
    seq_q: [batch, seq_len]
    seq_k: [batch, seq_len]

    return:
    mask: [batch, len_q, len_k]

    """
    # 获取
    batch, len_q = seq_q.size()
    batch, len_k = seq_k.size()

    # 假设<PAD>在字典中定义索引为0
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch, 1, len_k]

    return pad_attn_mask.expand(batch, len_q, len_k)  # [batch, len_q, len_k]


# Subsequent Mask for Decoder  为了防止Decoder的自回归信息泄露而生的Mask, 直接生成一个上三角矩阵
def get_attn_subsequent_mask(seq):
    """
    Build attention mask matrix for decoder when it autoregressing.

    parameters:
    seq: [batch, target_len]

    return:
    subsequent_mask: [batch, target_len, target_len]

    """
    attn_shape = [seq.shape[0], seq.shape[1], seq.shape[1]]  # [batch, target_len, target_len]

    # 生成一个上三角矩阵
    subsequent_mask = np.triu(np.zeros(attn_shape), k=1)  # [batch, target_len, target_len]
    subsequent_mask = torch.from_numpy(subsequent_mask)

    return subsequent_mask  # [batch, target_len, target_len].


# Positional Encoding   绝对位置编码, 用于传输给模型Self - Attention所不能传输的位置信息
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, droupout=0.1, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=p_drop)

        positional_encoding = torch.zeros(max_len, d_model)  # [max_len, d_model]

        # 创建一个长度为max_len的下标张量
        position = torch.arange(0, max_len).float().unsqueeze(1)  # [max_len, 1]  下标[0, 1, 2, ..., max_len]

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-torch.log(torch.Tensor([10000])) / d_model))  # [d_model / 2]

        # pos(i, 2j) 下标为偶数
        positional_encoding[:, 0::2] = torch.sin(position * div_term)  # even
        # pos(i, 2j + 1) 下标为奇数
        positional_encoding[:, 1::2] = torch.cos(position * div_term)  # odd

        # [max_len, d_model] --> [1, max_len, d_model] --> [max_lem, 1, d_model]
        positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)

        # register pe to buffer and require no grads
        self.register_buffer('pe', positional_encoding)

    def forward(self, x):
        """
        x: [seq_len, batch, d_model]
        we can add positional encoding to x directly, and ignore other dimension
        """
        # self.pe: [max_len, 1, d_model]
        x = x + self.pe[x.size(0), ...]

        return self.dropout(x)


# Feed Forward Neural Network 使用1*1卷积或者线性层
class FeedForwardNetwork(nn.Module):
    """
    Using nn.Conv1D raplace nn.Linear to implements FFN.
    """

    def __init__(self):
        super(FeedForwardNetwork, self).__init__()
        # dimension: d_model --> d_ff --> d_model
        # self.ff1 = nn.Linear(d_model, d_ff)
        # self.ff2 = nn.Linear(d_ff, d_model)
        self.ff1 = nn.Conv1d(d_model, d_ff, 1)  # Conv1d inputs: [batch, d_model, seq_len]
        self.ff2 = nn.Conv1d(d_ff, d_model, 1)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=p_drop)
        self.layer_norm = nn.LayerNorm(d_model)  # d_model
        self.layer_norm = nn.LayerNorm(d_model)  # d_model

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        residual = x
        x = x.transpose(1, 2)  # [batch, seq_len, d_model] --> [batch, d_model, seq_len]

        x = self.ff1(x)
        x = self.relu(x)
        x = self.ff2(x)

        x = x.transpose(1, 2)  # [batch, d_model, seq_len] --> [batch, seq_len, d_model]

        return self.layer_norm(x + residual)


# Scaled DotProduct Attention   点积注意力
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        Q:  [batch, n_heads, len_q, d_q]
        K:  [batch, n_heads, len_k, d_k]
        V:  [batch, n_heads, len_v, d_v]
        attn_mask:  [batch, n_heads, seq_len, seq_len]
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # [batch, n_heads, len_q, len_k]

        # masked_fill_能把传进来的Mask为True的地方全都填充上某个值, 这里需要用一个很大的负数来保证其在softmax后值为0
        # 经过这一步操作，scores中被mask掉的值经过softmax后，值会变得非常小
        scores.masked_fill_(attn_mask, -14)

        attn = nn.Softmax(dim=-1)(scores)  # [batch, n_heads, len_q, len_k]  注意力分数
        prob = torch.matmul(attn, V)  # [batch, n_heads, len_q, d_v]

        return prob, attn


# Multi-Head Attention  多头自注意力 每个注意力头只倾向于关注某一方面的语义相似性,使用不同的头来获取不同的特征
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        # do not use more instance to implement multihead attention
        # it can be complete in one matrix
        self.n_heads = n_heads

        # we can't use bias because there is no bias term in formular
        self.W_Q = nn.Linear(d_model, d_k * self.n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * self.n_heads, bias=False)
        self.fc = nn.Linear(d_v * self.n_heads, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        To make sure multihead attention can be used both in encoder and decoder,
        we ues Q, K, V respectively.
        input_Q: [batch, len_q, d_model]
        input_K: [batch, len_k, d_model]
        input_V: [batch, len_v, d_model]
        """
        residual, batch = input_Q, input_Q.size(0)
        # [batch, len_q, d_model] -- matmul W_Q --> [batch, len_q, d_q * n_heads] -- view -->
        # [batch, len_q, n_heads, d_k] -- transpose --> [batch, n_heads, len_q, d_k]

        Q = self.W_Q(input_Q).view(batch, -1, n_heads, d_k).transpose(1, 2)  # [batch, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch, -1, n_heads, d_k).transpose(1, 2)  # [batch, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch, -1, n_heads, d_v).transpose(1, 2)  # [batch, n_heads, len_v, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # [batch, n_heads, seq_len, seq_len]

        # prob: [batch, n_heads, len_q, d_v] attn: [batch, n_heads, len_q, len_k]
        prob, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)

        prob = prob.transpose(1, 2).contiguous()  # [batch, len_q, n_heads, d_v]
        prob = prob.view(batch, -1, n_heads * d_v).contiguous()  # [batch, len_q, n_heads * d_v]

        output = self.fc(prob)  # [batch, len_q, d_model]

        return self.layer_norm(residual + output), attn


# Encoder Attention + FFN
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.encoder_self_attn = MultiHeadAttention()
        self.ffn = FeedForwardNetwork()

    def forward(self, encoder_input, encoder_pad_mask):
        """
        encoder_input: [batch, source_len, d_model]
        encoder_pad_mask:   [batch, n_heads, source_len, source_len]

        encode_output: [batch, source_len, d_model]
        attn:   [batch, n_heads, source_len, source_len]
        """
        encoder_output, attn = self.encoder_self_attn(input_Q=encoder_input, input_K=encoder_input,
                                                      input_V=encoder_input, attn_mask=encoder_pad_mask)
        encoder_output = self.ffn(encoder_output)  # [batch, source_len, d_model]

        return encoder_output, attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.source_embedding = nn.Embedding(source_vocab_size, d_model)
        self.positional_embedding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for layer in range(n_layers)])

    def forward(self, encoder_input):
        # encoder_input: [batch, source_len]
        encoder_output = self.source_embedding(encoder_input)  # [batch, source_len, d_model]
        # [batch, source_len, d_model]
        encoder_output = self.positional_embedding(encoder_output.transpose(0, 1)).transpose(0, 1)

        encoder_self_attn_mask = get_attn_pad_mask(encoder_input, encoder_input)  # [batch, source_len, source_len]
        encoder_self_attns = list()

        for layer in self.layers:
            # encoder_output: [batch, source_len, d_model]
            # encoder_attn_mask: [batch, n_heads, source_len, source_len]
            encoder_output, encoder_self_attn = layer(encoder_output, encoder_self_attn_mask)
            encoder_self_attns.append(encoder_self_attn)

        return encoder_output, encoder_self_attns


# Decoder
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.decoder_self_attn = MultiHeadAttention()
        self.encoder_decoder_attn = MultiHeadAttention
        self.ffn = FeedForwardNetwork()

    def forward(self, decoder_input, encoder_output, decoder_self_mask, decoder_encoder_mask):
        """
        decoder_input: [batch, source_len, d_model]
        encoder_input: [batch, source_len, d_model]
        decoder_self_mask: [batch, target_len, target_len]
        decoder_encoder_mask: [batch, target_len, source_len]
        """
        # masked mutlihead attention
        # Q, K, V all from decoder it self
        # decoder_output: [batch, target_len, d_model]
        # decoder_self_attn: [batch, n_heads, target_len, target_len]
        decoder_output, decoder_self_attn = self.decoder_self_attn(decoder_input, decoder_input, decoder_input,
                                                                   decoder_self_mask)

        # Q from decoder, K, V from encoder
        # decoder_output: [batch, target_len, d_model]
        # decoder_encoder_attn: [batch, n_heads, target_len, source_len]
        decoder_output, decoder_encoder_attn = self.encoder_decoder_attn(decoder_input, encoder_output, encoder_output,
                                                                         decoder_encoder_mask)
        decoder_output = self.ffn(decoder_output)  # [batch, target_len, d_model]

        return decoder_output, decoder_self_attn, decoder_encoder_attn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.target_embedding = nn.Embedding(target_vocab_size, d_model)
        self.positional_embedding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for layer in range(n_layers)])

    def forward(self, decoder_input, encoder_input, encoder_output):
        """
        decoder_input: [batch, target_len]
        encoder_input: [batch, source_len]
        encoder_output: [batch, source_len, d_model]
        """
        decoder_output = self.target_embedding(decoder_input)  # [batch, target_len, d_model]
        # [batch, target_len, d_model]
        decoder_output = self.positional_embedding(decoder_output.transpose(0, 1)).transpose(0, 1)

        decoder_self_attn_mask = get_attn_pad_mask(decoder_input, decoder_input)  # [batch, target_len, target_len]
        decoder_subsequent_mask = get_attn_subsequent_mask(decoder_input)  # [batch, target_len, target_len]

        decoder_encoder_attn_mask = get_attn_pad_mask(decoder_input, encoder_input)  # [batch, target_len, source_len]

        decoder_self_mask = torch.gt(decoder_self_attn_mask + decoder_subsequent_mask, 0)
        decoder_self_attns, decoder_encoder_attns = [], []

        for layer in self.layers:
            # decoder_output: [batch, target_len, d_model]
            # decoder_self_attn: [batch, n_heads, target_len, target_len]
            # decoder_encoder_attn: [batch, n_heads, target_len, source_len]
            decoder_output, decoder_self_attn, decoder_encoder_attn = layer(decoder_output, encoder_output,
                                                                            decoder_self_mask,
                                                                            decoder_encoder_attn_mask)
            decoder_self_attns.append(decoder_self_attn)
            decoder_encoder_attns.append(decoder_encoder_attn)

        return decoder_output, decoder_self_attns, decoder_encoder_attns


# Transformer
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, target_vocab_size, bias=False)

    def forward(self, encoder_input, decoder_input):
        """
        encoder_input: [batch, source_len]
        decoder_input: [batch, target_len]
        """
        # encoder_output: [batch, source_len, d_model]
        # encoder_attns: [n_layers, batch, n_heads, source_len, source_len]
        encoder_output, encoder_attns = self.encoder(encoder_input)
        # decoder_output: [batch, target_len, d_model]
        # decoder_self_attns: [n_layers, batch, n_heads, target_len, target_len]
        # decoder_encoder_attns: [n_layers, batch, n_heads, target_len, source_len]
        decoder_output, decoder_self_attns, decoder_encoder_attns = self.decoder(decoder_input, encoder_input,
                                                                                 encoder_output)
        decoder_logits = self.projection(decoder_output)  # [batch, target_len, target_vocab_size]

        # decoder_logits: [batch * target_len, target_vocab_size]
        return decoder_logits.view(-1, decoder_logits.size(-1)), encoder_attns, decoder_self_attns, \
            decoder_encoder_attns

