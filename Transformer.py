import torch
import torch.nn as nn
import decoder as dec
import encoder as enc
class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, vocab_size):
        super(Transformer, self).__init__()


        self.encoder = enc.Encoder(d_model, num_heads, num_layers)
        self.decoder = dec.Decoder(d_model, num_heads, num_layers, vocab_size)
    def forward(self, enc_input, dec_input, encoder_padding_mask, decoder_lookahead_mask, decoder_padding_mask):
        decoder_masked_attention_mask = decoder_lookahead_mask + decoder_padding_mask
        decoder_cross_attention_mask = decoder_padding_mask
        enc_out = self.encoder(enc_input, encoder_padding_mask)
        dec_out = self.decoder(dec_input, enc_out, decoder_masked_attention_mask, decoder_cross_attention_mask)
        return dec_out
