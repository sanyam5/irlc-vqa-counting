from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from config import *
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F


class QuestionParser(nn.Module):
    word_dim = 300
    ques_dim = 1024
    glove_file = DATA_DIR + "/glove6b_init_300d.npy"

    def __init__(self, dropout=0.1):
        super(QuestionParser, self).__init__()
        self.embd = nn.Embedding(VOCAB_SIZE + 1, self.word_dim, padding_idx=VOCAB_SIZE)
        self.rnn = nn.GRU(self.word_dim, self.ques_dim)
        self.dropout = nn.Dropout(dropout)
        self.glove_init()

    def glove_init(self):
        print("initialising with glove embeddings")
        glove_embds = torch.from_numpy(np.load(self.glove_file))
        assert glove_embds.size() == (VOCAB_SIZE, self.word_dim)
        self.embd.weight.data[:VOCAB_SIZE] = glove_embds
        print("done.")

    def forward(self, questions):
        # (B, MAXLEN)
        # print("question size ", questions.size())
        questions = questions.t()  # (MAXLEN, B)
        questions = self.embd(questions)  # (MAXLEN, B, word_size)
        _, q_emb = self.rnn(questions)
        q_emb = q_emb[-1]  # (B, ques_size)
        q_emb = self.dropout(q_emb)

        return q_emb


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(nn.ReLU())
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(nn.ReLU())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ScoringFunction(nn.Module):
    v_dim = 2048
    q_dim = QuestionParser.ques_dim
    score_dim = 512

    def __init__(self, dropout=0.1):
        super(ScoringFunction, self).__init__()
        self.v_drop = nn.Dropout(dropout)
        self.q_drop = nn.Dropout(dropout)
        self.v_proj = FCNet([self.v_dim, self.score_dim])
        self.q_proj = FCNet([self.q_dim, self.score_dim])
        self.dropout = nn.Dropout(dropout)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        batch, k, _ = v.size()
        v = self.v_drop(v)
        q = self.q_drop(q)
        v_proj = self.v_proj(v)  # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)  # [batch, k, qdim]
        s = v_proj * q_proj
        s = self.dropout(s)
        return s  # (B, k, score_dim)


class SoftCount(nn.Module):

    def __init__(self):
        super(SoftCount, self).__init__()
        self.ques_parser = QuestionParser()
        self.f = ScoringFunction()
        self.W = weight_norm(nn.Linear(self.f.score_dim, 1), dim=None)

    def forward(self, v_emb, q):
        # v_emb = (B, k, v_dim)
        # q = (B, MAXLEN)

        q_emb = self.ques_parser(q)  # (B, q_dim)
        s = self.f(v_emb, q_emb)  # (B, k, score_dim)
        soft_counts = F.sigmoid(self.W(s)).squeeze(2)  # (B, k)
        C = soft_counts.sum(dim=1)  # (B,)
        return C
