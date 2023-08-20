import torch
import torch.nn as nn
import multihead_attention as ma
import feedforward as ff
import numpy as np
def positional_encoding_mat(batch_dim, max_seq_len, d_model, n = 10000):
    positional_mat = torch.zeros(max_seq_len, d_model)
    cumul = []
    for batch in range(batch_dim):
        for i in range(max_seq_len):
            for d in range(d_model):
                if (d == 0):
                    denominator = 1
                else:
                    denominator = pow(n, (d)/d_model)
                if (d % 2 == 0):
                    positional_mat[i][d] = np.sin(i / denominator)
                else:
                    denominator = pow(n, (d-1)/d_model)
                    positional_mat[i][d] = np.cos(i/ denominator)
        cumul.append(positional_mat)
    cum = torch.stack(cumul)
    return cum

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super(EncoderBlock, self).__init__()
        self.num_heads = num_heads
        self.multi_head_attention = ma.MultiHeadAttention(d_model, num_heads)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ffn = ff.FeedForward(d_model)
    def forward(self, x, padding_mask):
        residual_x1 = x
        self_aware = self.multi_head_attention(x, padding_mask)
        x2 = self.layer_norm1(residual_x1 + self_aware)
        residual_x2 = x2
        ffn_out = self.ffn(x2)
        out = self.layer_norm2(residual_x2 + ffn_out)
        return out
class EncoderSequential(nn.Sequential):
    '''
    If we do not override forward() and let Decoder(x, mask) do the job, it will fail.
    Since we are using the sequential model, nn.module.Sequential would not know which argument to choose
    to run forward(x) when called Decoder(x, mask).
    '''
    def forward(self, *inputs):
        enc_input, mask = inputs
        for module in self._modules.values():
            y = module(enc_input, mask)
        return y
class Encoder(nn.Module):
    def __init__(self,d_model, num_heads, layer_num):
        super(Encoder, self).__init__()
        self.encoder_layers = []
        for _ in range(layer_num):
            self.encoder_layers.append(EncoderBlock(d_model, num_heads))
        self.seq_encoder_blocks = EncoderSequential(*self.encoder_layers)
    def forward(self, x, padding_mask):
        batch_dim, max_seq_len, d_model = x.size()
        positional_mat = positional_encoding_mat(batch_dim, max_seq_len, d_model)
        x = x + positional_mat
        x = self.seq_encoder_blocks(x, padding_mask)
        return x





