import torch
import torch.nn as nn
import numpy as np

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_head):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_head = num_head
        self.d_e = d_model
        self.head_dim = self.d_e // self.num_head
        self.W_qk = nn.Linear(self.d_e, self.d_e * 2)
        self.linear = nn.Linear(self.d_e, self.d_e)
    def forward(self, dec_input, enc_input, mask=None):
        batch_size, seq_len, d_model = dec_input.size()
        mixed_qk = self.W_qk.forward(enc_input)
        mixed_qkv = torch.cat((mixed_qk, dec_input), dim = 2)
        mixed_qkv = torch.vsplit(mixed_qkv, batch_size)

        mixed_qkv = [torch.hsplit(torch.squeeze(example), self.num_head) for example in mixed_qkv]
        example_Q, example_K, example_V = [], [], []
        for example in mixed_qkv:
            Q_list, K_list, V_list = [], [], []
            for head in example:
                q, k, v = torch.hsplit(head, 3)
                Q_list.append(q), K_list.append(k), V_list.append(v)
            example_Q.append(torch.stack(Q_list)), example_K.append(torch.stack(K_list)), example_V.append(torch.stack(V_list))

        Q = torch.stack(example_Q)
        K = torch.stack(example_K)
        V = torch.stack(example_V)

        atten_score = torch.matmul(Q, K.transpose(-1, -2))
        if (mask != None):
            masking = float('-10e10') * mask
            atten_score = atten_score + masking
        atten_score = atten_score / np.sqrt(self.head_dim)
        F = torch.nn.Softmax(-1)

        atten_score = F(atten_score)
        out = torch.matmul(atten_score, V)
        values = torch.reshape(out, (batch_size, seq_len, d_model))
        values = self.linear(values)
        return values


