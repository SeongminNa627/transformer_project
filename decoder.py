import torch
import torch.nn as nn
import multihead_attention as ma
import multihead_cross_attention as mca
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

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super(DecoderBlock, self).__init__()
        self.num_heads = num_heads
        self.masked_multihead_attention = ma.MultiHeadAttention(d_model, num_heads)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.multihead_cross_attention = mca.MultiHeadCrossAttention(d_model, num_heads)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ffn = ff.FeedForward(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
    def forward(self, dec_input, enc_input, masked_attention_mask, cross_attention_mask):
        residual_x1 = dec_input
        masked_out = self.masked_multihead_attention(dec_input, masked_attention_mask)
        add_norm1 = self.layer_norm1(residual_x1 + masked_out)
        residual_x2 = add_norm1
        cross_out = self.multihead_cross_attention(add_norm1, enc_input, cross_attention_mask)
        add_norm2 = self.layer_norm2(residual_x2 + cross_out)
        residual_x3 = add_norm2
        fc_out = self.ffn(add_norm2)
        add_norm3 = self.layer_norm3(residual_x3 + fc_out)
        return add_norm3
class DecoderSequential(nn.Sequential):
    '''
    If we do not override forward() and let Decoder(x, mask) do the job, it will fail.
    Since we are using the sequential model, nn.module.Sequential would not know which argument to choose
    to run forward(x) when called Decoder(x, mask).
    '''
    def forward(self, *inputs):
        dec_input, enc_input, masked_attention_mask, cross_attention_mask = inputs
        for module in self._modules.values():
            y = module(dec_input, enc_input, masked_attention_mask, cross_attention_mask)
        return y
class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, layer_num, vocab_size):
        super(Decoder, self).__init__()
        self.decoder_layers = []
        for _ in range(layer_num):
            self.decoder_layers.append(DecoderBlock(d_model, num_heads))
        self.seq_decoder_blocks = DecoderSequential(*self.decoder_layers)
        self.linear = nn.Linear(d_model, vocab_size)
    def forward(self, dec_input, enc_input, masked_attention_mask, cross_attention_mask):
        batch_dim, max_seq_len, d_model = dec_input.size()
        positional_mat = positional_encoding_mat(batch_dim, max_seq_len, d_model)
        dec_input = dec_input + positional_mat
        x = self.seq_decoder_blocks(dec_input, enc_input, masked_attention_mask, cross_attention_mask)
        dec_out = self.linear(x)
        return dec_out




