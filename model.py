#!/usr/bin/env python

import numpy as np
from numpy.random import permutation, randint

import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

torch.manual_seed(1)  # cpu
torch.cuda.manual_seed(1)  # gpu
np.random.seed(1)  # numpy
torch.backends.cudnn.deterministic = True  # cudnn


class AspectMemoryEncoder(nn.Module):

    def __init__(self, vocab_size, emb_size, w_emb, a_emb, fix_w_emb=True, extend_aspect=0):

        super(AspectMemoryEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.num_aspects = a_emb.shape[0]
        self.extend_aspect = extend_aspect

        self.lookup = nn.Embedding(vocab_size, emb_size)
        if w_emb is None:
            xavier_uniform_(self.lookup.weight.data)
        else:
            assert w_emb.size() == (vocab_size, emb_size), "Word embedding matrix has incorrect size"
            self.lookup.weight.data.copy_(w_emb)
            self.lookup.weight.requires_grad = not fix_w_emb

        self.register_buffer('a_emb1', a_emb)

        if extend_aspect:
            self.a_emb2 = nn.Parameter(torch.Tensor(extend_aspect, emb_size))
            xavier_uniform_(self.a_emb2.data)
            self.num_aspects += extend_aspect

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        # inputs: batch_size * sent_len
        # inputs_2: ? * sent_len

        if self.extend_aspect:
            a_emb = torch.cat((self.a_emb1, self.a_emb2), dim=0)
        else:
            a_emb = self.a_emb1

        enc_output, enc_attention = self.encode_input(inputs, a_emb)  # batch_size*emb_size

        return enc_output, enc_attention

    def encode_input(self, inputs, a_emb):
        x_wrd = self.lookup(inputs)  # batch_size * sent_len * emb_size
        sim = F.cosine_similarity(a_emb.unsqueeze(1).unsqueeze(1), x_wrd.unsqueeze(0), dim=-1)  # (a,1,1,e) * (1,b,s,e) -> a,b,s
        # sim = sim.masked_fill(sim == 0, -1e9)
        sim, _ = sim.max(dim=0)  # b,s

        a = F.softmax(sim, dim=1).unsqueeze(-1)  # b,s,1
        z = a.transpose(1, 2).matmul(x_wrd)
        z = z.squeeze() #b, e

        if z.dim() == 1:
            return z.unsqueeze(0), a.squeeze(-1)  
        return z, a.squeeze(-1) #(b,e) and (b,s)

    def predict_by_aspect(self, inputs, softmax=False):
        k, a, emb = 0, 0, 0
        if inputs.dim() == 3:
            k, a, emb = inputs.size()
            inputs = inputs.reshape(-1, emb)  # k*a, e
        aspects = self.get_aspects()  # a,e

        inputs = inputs.unsqueeze(1).expand(-1, aspects.size(0), -1)  # k*a, a, e
        aspects = aspects.unsqueeze(0).expand(inputs.size(0), -1, -1)  # k*a, a, e
        score = F.cosine_similarity(aspects, inputs, dim=2)  # k*a, a
        if softmax:
            score = self.softmax(score)  # k*a, a
        if k != 0:
            return score.reshape(k, a, -1)
        return score

    def get_aspects(self):
        if self.extend_aspect:
            return torch.cat((self.a_emb1, self.a_emb2), dim=0)
        else:
            return self.a_emb1


class AspectMemorySummarizer(nn.Module):

    def __init__(self, vocab_size, emb_size,
                 w_emb=None, a_emb=None, a_weight=None,
                 fix_w_emb=True, word_thres=0.2):

        super(AspectMemorySummarizer, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size

        self.lookup = nn.Embedding(vocab_size, emb_size)
        if w_emb is None:
            xavier_uniform_(self.lookup.weight.data)
        else:
            assert w_emb.size() == (vocab_size, emb_size), "Word embedding matrix has incorrect size"
            self.lookup.weight.data.copy_(w_emb)
            self.lookup.weight.requires_grad = not fix_w_emb

        self.register_buffer('a_emb', a_emb)
        self.register_buffer('a_weight', torch.from_numpy(np.array(a_weight, dtype='float32')))

        self.word_sim_thres = nn.Threshold(word_thres, 0)
        self.sim_thres = nn.Threshold(0, 0)

    def forward(self, inputs):
        # inputs: batch_size * sent_len
        # a_emb: asp_num * emb_size

        # import pdb
        # pdb.set_trace()

        inputs_len = (inputs != 0).sum(1).float()  # batch_size * 1
        x_wrd = self.lookup(inputs)  # batch_size * sent_len * emb_size
        sim = F.cosine_similarity(self.a_emb.unsqueeze(1).unsqueeze(1), x_wrd.unsqueeze(0), dim=-1)  # a,b,s
        sim = self.word_sim_thres(sim)
        sim = sim * self.a_weight.unsqueeze(1).unsqueeze(1)

        self.centroid_score, self.keyword_idx = self.sim_thres(sim).max(dim=0)  # b,s
        centroid_score = self.centroid_score.sum(dim=1)  # b
        centroid_score = centroid_score / (inputs_len + 1e-5)

        sim = sim.masked_fill(sim == 0, -1e9)
        sim, _ = sim.max(dim=0)  # b,s

        a = F.softmax(sim, dim=1).unsqueeze(-1)  # b,s,1
        z = a.transpose(1, 2).matmul(x_wrd)  # b,1,e

        enc_out = z.squeeze(1)  # b,e
        enc_out = enc_out * (centroid_score > 0.0001).float().unsqueeze(1)
        enc_attention = a.squeeze(-1)  # b,s
        return enc_out, enc_attention, centroid_score

    def get_aspects(self):
        return self.a_emb

    def get_attention_info(self):
        return self.centroid_score.data.cpu().numpy(), self.keyword_idx.data.cpu().numpy()


class OrthogonalityLoss(Module):
    def forward(self, input):
        inp_n = input / input.norm(p=2, dim=1, keepdim=True)
        eye = torch.eye(inp_n.size(0)).type_as(inp_n)
        loss = torch.sum((inp_n.matmul(inp_n.t()) - eye) ** 2)
        return loss


def partial_softmax(logits, weights, dim):
    exp_logits = torch.exp(logits)  # b * s * r
    if len(exp_logits.size()) == len(weights.size()):  # b * s
        exp_logits_weighted = torch.mul(exp_logits, weights)  # b * s
    else:
        exp_logits_weighted = torch.mul(exp_logits, torch.unsqueeze(weights, -1))  # b * s * r
    exp_logits_sum = torch.sum(exp_logits_weighted, dim=dim, keepdim=True)  # b * 1 * (r)
    partial_softmax_score = torch.div(exp_logits_weighted, exp_logits_sum)
    return partial_softmax_score
