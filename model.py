from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from config import *
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F
from torch.autograd import Variable


class QuestionParser(nn.Module):
    word_dim = 300
    ques_dim = 1024
    glove_file = DATA_DIR + "/glove6b_init_300d.npy"

    def __init__(self, dropout=0.3):
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
        _, (q_emb) = self.rnn(questions)
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
            layers.append(nn.LeakyReLU())
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(nn.LeakyReLU())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ScoringFunction(nn.Module):
    v_dim = 2048
    q_dim = QuestionParser.ques_dim
    score_dim = 1024

    def __init__(self, dropout=0.3):
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


class GTUScoringFunction(nn.Module):
    v_dim = 2048
    q_dim = QuestionParser.ques_dim
    score_dim = 2048

    def __init__(self, dropout=0.1):
        super(GTUScoringFunction, self).__init__()
        self.dense1 = nn.Linear(self.v_dim + self.q_dim, self.score_dim)
        self.dense2 = nn.Linear(self.v_dim + self.q_dim, self.score_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        batch, k, _ = v.size()

        q = q[:, None, :].repeat(1, k, 1)  # (B, k, q_dim)
        vq = torch.cat([v, q], dim=2)  # (B, k, v_dim + q_dim)

        y = F.tanh(self.dense1(vq))  # (B, k, score_dim)
        g = F.sigmoid(self.dense2(vq))  # (B, k, score_dim)

        s = y * g
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


class RhoScorer(nn.Module):

    def __init__(self):
        super(RhoScorer, self).__init__()
        self.W = weight_norm(nn.Linear(QuestionParser.ques_dim, 1), dim=None)

        inp_dim = 1 + 1 + 6 + 6 + 1 + 1 + 1  # 17
        self.f_rho = FCNet([inp_dim, 100])
        self.dense = weight_norm(nn.Linear(100, 1), dim=None)

    @staticmethod
    def get_spatials(b):
        # b = (B, k, 6)

        b = b.float()

        B, k, _ = b.size()

        b_ij = torch.stack([b] * k, dim=1)  # (B, k, k, 6)
        b_ji = b_ij.transpose(1, 2)

        area_ij = (b_ij[:, :, :, 2] - b_ij[:, :, :, 0]) * (b_ij[:, :, :, 3] - b_ij[:, :, :, 1])
        area_ji = (b_ji[:, :, :, 2] - b_ji[:, :, :, 0]) * (b_ji[:, :, :, 3] - b_ji[:, :, :, 1])

        righmost_left = torch.max(b_ij[:, :, :, 0], b_ji[:, :, :, 0])
        downmost_top = torch.max(b_ij[:, :, :, 1], b_ji[:, :, :, 1])
        leftmost_right = torch.min(b_ij[:, :, :, 2], b_ji[:, :, :, 2])
        topmost_down = torch.min(b_ij[:, :, :, 3], b_ji[:, :, :, 3])

        # calucate the separations
        left_right = (leftmost_right - righmost_left)
        up_down = (topmost_down - downmost_top)

        # don't multiply negative separations,
        # might actually give a postive area that doesn't exit!
        left_right = torch.max(0*left_right, left_right)
        up_down = torch.max(0*up_down, up_down)

        overlap = left_right * up_down

        iou = overlap / (area_ij + area_ji - overlap)
        o_ij = overlap / area_ij
        o_ji = overlap / area_ji

        iou = iou.unsqueeze(3)  # (B, k, k, 1)
        o_ij = o_ij.unsqueeze(3)  # (B, k, k, 1)
        o_ji = o_ji.unsqueeze(3)  # (B, k, k, 1)

        return b_ij, b_ji, iou, o_ij, o_ji

    def forward(self, q_emb, v_emb, b):
        # q_emb = (B, ques_size)
        # v_emb = (B, k, v_dim)
        # b = (B, k, 6)

        B, k, _ = v_emb.size()

        features = []

        wq = self.W(q_emb).squeeze(1)  # (B,)
        wq = wq[:, None, None, None].repeat(1, k, k, 1)  # (B, k, k, 1)
        assert wq.size() == (B, k, k, 1), "wq size is {}".format(wq.size())
        features.append(wq)

        norm_v_emb = F.normalize(v_emb, dim=2)  # (B, k, v_dim)
        vtv = torch.bmm(norm_v_emb, norm_v_emb.transpose(1, 2))  # (B, k, k)
        vtv = vtv[:, :, :, None].repeat(1, 1, 1, 1)  # (B, k, k, 1)
        assert vtv.size() == (B, k, k, 1)
        features.append(vtv)

        b_ij, b_ji, iou, o_ij, o_ji = self.get_spatials(b)

        assert b_ij.size() == (B, k, k, 6)
        assert b_ji.size() == (B, k, k, 6)
        assert iou.size() == (B, k, k, 1)
        assert o_ij.size() == (B, k, k, 1)
        assert o_ji.size() == (B, k, k, 1)

        features.append(b_ij)  # (B, k, k, 6)
        features.append(b_ji)  # (B, k, k, 6)
        features.append(iou)  # (B, k, k, 1)
        features.append(o_ij)  # (B, k, k, 1)
        features.append(o_ji)  # (B, k, k, 1)

        features = torch.cat(features, dim=3)  # (B, k, k, 17)

        rho = self.f_rho(features)  # (B, k, k, 100)
        rho = self.dense(rho).squeeze(3)  # (B, k, k)

        return rho, features  # (B, k, k)


class IRLC(nn.Module):

    def __init__(self):
        super(IRLC, self).__init__()
        self.ques_parser = QuestionParser()
        self.f_s = ScoringFunction()
        self.W = weight_norm(nn.Linear(self.f_s.score_dim, 1), dim=None)
        self.f_rho = RhoScorer()

        # extra custom parameters
        self.eps = nn.Parameter(torch.zeros(1))
        self.extra_params = nn.ParameterList([self.eps])
        self.EMA = 0

    def sample_action(self, probs, already_selected=None, greedy=False):
        # probs = (B, k+1)
        # already_selected = (num_timesteps, B)

        if already_selected is None:
            mask = 1
        else:
            mask = Variable(torch.ones(probs.size()))
            if USE_CUDA:
                # TODO: uncomment this, when this model works
                mask = mask.cuda()
                pass
            mask = mask.scatter_(1, already_selected.t(), 0)  # (B, k+1)

        masked_probs = mask * (probs + 1e-20)  # (B, k+1), add epsilon to make sure no non-masked value is zero.
        rescaled_masked_probs = masked_probs / (1e-20 + masked_probs.sum(dim=1)[:, None])

        if greedy:
            _, a = rescaled_masked_probs.max(dim=1)  # (B)
        else:
            a = rescaled_masked_probs.multinomial(num_samples=1).squeeze(dim=1)  # (B)

        return a, rescaled_masked_probs

    @staticmethod
    def get_interaction(rho, a):
        # get the interaction row in rho corresponding to the action a
        # rho = (B, num_actions, k)
        # a = (B) containing action indices between 0 and num_actions-1

        B, _, k = rho.size()

        # first expand a to the size required output
        a = a[:, None].repeat(1, k)  # (B, k)

        # print("rho size = {} and a size = {}".format(rho.size(), a.size()))
        interaction = rho.gather(dim=1, index=a.unsqueeze(dim=1)).squeeze(dim=1)  # (B, k)
        assert interaction.size() == (B, k), "interaction size is {}".format(interaction.size())
        # print("interaction size = {}".format(interaction.size()))

        return interaction  # (B, k)

    def sample_objects(self, kappa_0, rho, batch_eps, greedy=False):
        # kappa_0 = (B, k)
        # rho = (B, k, k)

        # add an extra row of 0 interaction for the terminal action
        rho = torch.cat((rho, 0 * rho[:, :1, :]), dim=1)  # (B, k+1, k)

        B, k = kappa_0.size()

        P = None  # will store probability values at each timestep.
        A = None  # will store action values at each timestep.
        T = k+1  # num timesteps = different possible actions. +1 for the terminal action
        kappa = kappa_0  # (B, k), starting kappa

        for t in range(T):
            # calculate probabilities of each action
            unscaled_p = F.softmax(torch.cat((kappa, batch_eps), dim=1), dim=1)  # (B, k+1)
            # print("p = ", p)
            # select one object (called "action" in RL terms), avoid already selected objects.
            a, scaled_p = self.sample_action(probs=unscaled_p, already_selected=A, greedy=greedy)  # (B)
            # update kappa logits with the row in the interaction matrix corresponding to the chosen action.
            interaction = self.get_interaction(rho, a)
            kappa = kappa + interaction

            # record the prob and action values at each timestep for later use
            P = scaled_p[None] if P is None else torch.cat((P, scaled_p[None]), dim=0)  # (t+1, B, k+1)
            A = a[None] if A is None else torch.cat((A, a[None]), dim=0)  # (t+1, B)

        assert P.size() == (T, B, k+1)
        assert A.size() == (T, B)

        # calculate count
        terminal_action = (A == k)  # (T, B)  # true for the timestep when terminal action was selected.
        _, count = terminal_action.max(dim=0)  # (B,)  # index of the terminal action is considered the count

        return P, A, count

    def compute_vars(self, v_emb, b, q):
        # v_emb = (B, k, v_dim)
        # b = (B, k, 6)
        # q = (B, MAXLEN)

        B, k, _ = v_emb.size()

        q_emb = self.ques_parser(q)  # (B, q_dim)
        s = self.f_s(v_emb, q_emb)  # (B, k, score_dim)
        kappa_0 = self.W(s).squeeze(2)  # (B, k)

        rho, _ = self.f_rho(q_emb, v_emb, b)  # (B, k, k)

        return kappa_0, rho

    def take_mc_samples(self, kappa_0, rho, num_mc_samples):
        # kappa_0 = (B, k)
        # rho = (B, k, k)

        B, k = kappa_0.size()
        assert rho.size() == (B, k, k)

        kappa_0 = kappa_0.repeat(num_mc_samples, 1)  # (B * samples, k)
        rho = rho.repeat(num_mc_samples, 1, 1)  # (B * samples, k)

        batch_eps = torch.cat([self.eps] * B * num_mc_samples)[:, None]  # (B * samples, 1)

        P, A, count = self.sample_objects(kappa_0=kappa_0, rho=rho, batch_eps=batch_eps)
        _, _, greedy_count = self.sample_objects(kappa_0=kappa_0, rho=rho, batch_eps=batch_eps, greedy=True)

        return count, greedy_count, P, A, rho

    def get_sc_loss(self, count_gt, count, greedy_count, P, A):
        # count_gt = (B,)
        # count = (B,)
        # greedy_count = (B,)
        # P = (T, B, k+1)
        # A = (T, B)

        assert count.size() == count_gt.size()
        assert greedy_count.size() == count_gt.size()

        count = count.float()
        greedy_count = greedy_count.float()
        count_gt = count_gt.float()

        T, B, num_actions = P.size()

        # self-critical loss
        E = torch.abs(count - count_gt)  # (B,)
        E_greedy = torch.abs(greedy_count - count_gt)  # (B,)

        R = E_greedy - E  # (B,)
        # R = self.EMA - E  # (B,)

        # self.EMA = 0.99 * self.EMA + 0.01 * E.mean()

        # R = - E  # (B,)
        assert R.size() == count.size(), "R size is {}".format(R.size())

        terminal_A = (A == (num_actions-1)).float()  # (T, B)
        post_terminal_A = terminal_A.cumsum(dim=0) - terminal_A  # (T, B)
        valid_A = (A == A).float() - post_terminal_A  # (T, B)

        # we need to select the probabilities of actions sampled
        PA = P.gather(dim=2, index=A.unsqueeze(2)).squeeze(dim=2)  # (T, B)
        assert PA.size() == A.size()

        log_PA = torch.log(PA + 1e-20)  # (T, B)
        # sum_log_PA = log_PA.sum(dim=0)  # (B,)

        mean_log_PA = (log_PA * valid_A).sum(dim=0) / valid_A.sum(dim=0)

        batch_sc_loss = - R * mean_log_PA  # (B,)
        sc_loss = batch_sc_loss.mean(dim=0)  # (1,)

        return sc_loss

    def get_entropy_loss(self, P, A):
        # P = (T, B, k+1)
        # A = (T, B)

        T, B, num_actions = P.size()

        # P entropy loss
        def H(probs, dim):
            mults = probs * torch.log(probs + 1e-20)
            return - mults.sum(dim=dim)

        h = H(P, dim=2)  # (T, B)
        terminal_A = (A == (num_actions - 1)).float()  # (T, B)
        post_terminal_A = terminal_A.cumsum(dim=0) - terminal_A  # (T, B)
        valid_A = (A == A).float() - post_terminal_A  # (T, B)

        batch_entropy_loss = - (h * valid_A).sum(dim=0) / valid_A.sum(dim=0)  # (B,)

        entropy_loss = batch_entropy_loss.mean(dim=0)

        return entropy_loss

    def get_interaction_strength(self, rho):
        # rho = (B, k, k)

        B, k, _ = rho.size()

        # interaction strength, lower is better, sparse preferred.
        interaction_strengths = F.smooth_l1_loss(rho, 0*rho.detach(), reduce=False)  # (B, k, k)
        interaction_strengths = interaction_strengths.mean(dim=2)  # (B, k)
        # TODO: not sure if to be done only for selected actions. doing for all now.
        # selected_interactions = interactions[A]  # (T, B)
        # # sum along the time dimension
        batch_interaction_strength = interaction_strengths.sum(dim=1)  # (B,)

        batch_interaction_strength /= k

        interaction_strength = batch_interaction_strength.mean(dim=0)

        return interaction_strength

    def get_loss(self, count_gt, count, greedy_count, P, A, rho):
        # count_gt = (B,)
        # count = (B,)
        # greedy_count = (B,)
        # P = (T, B, k+1)
        # A = (T, B)
        # rho = (B, k, k)

        assert count.size() == count_gt.size()
        assert greedy_count.size() == count_gt.size()

        sc_loss = self.get_sc_loss(count_gt, count, greedy_count, P, A)
        entropy_loss = self.get_entropy_loss(P, A)
        interaction_strength = self.get_interaction_strength(rho)

        loss = 1.0 * sc_loss + .005 * entropy_loss + .005 * interaction_strength

        return loss
