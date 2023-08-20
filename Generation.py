import torch
import nltk
class Generation():
    def __init__(self, transformer, word_vec_model, MAX_SEQ_LEN = 50, D_MODEL = 304):
        self.transformer = transformer
        self.wv = word_vec_model.wv
        self.index_to_vector =  {i: self.wv[j] for i, j in enumerate(range(len(self.wv)))}
        self.vocab_list = self.wv.index_to.key
        self.MAX_SEQ_LEN  = MAX_SEQ_LEN
        self.D_MODEL = D_MODEL
        self.VOCAB_SIZE = len(self.wv)

        self.embed_SOS = torch.randn(1, D_MODEL).squeeze()
        self.index_SOS = self.VOCAB_SIZE
        self.index_to_vector[self.index_SOS] = self.embed_SOS
        self.vocab_list.append('<SOS>')
        self.VOCAB_SIZE = self.VOCAB_SIZE + 1

        self.embed_EOS = torch.randn(1, D_MODEL).squeeze()
        self.index_EOS = self.index_SOS + 1
        self.index_to_vector[self.index_EOS] = self.embed_EOS
        self.vocab_list.append('<EOS>')
        self.VOCAB_SIZE = self.VOCAB_SIZE + 1

        self.embed_UKN = torch.randn(1, D_MODEL).squeeze()
        self.index_UKN = self.index_EOS + 1
        self.index_to_vector[self.index_UKN] = self.embed_UKN
        self.vocab_list.append('<UKN>')
        self.VOCAB_SIZE = self.VOCAB_SIZE + 1

        self.embed_PAD = torch.randn(1, D_MODEL).squeeze()
        self.index_PAD = self.index_UKN + 1
        self.index_to_vector[self.index_PAD] = self.embed_PAD
        self.vocab_list.append('<PAD>')
        self.VOCAB_SIZE = self.VOCAB_SIZE + 1

    def tokenize(self, sentence, SOS=False, EOS=False):
        '''Given a sentence, return a vector representation (index) of the sentence'''
        tokenized = [self.index_PAD for _ in range(self.MAX_SEQ_LEN)]
        raw_tokens = nltk.word_tokenize(sentence)
        for i, token in enumerate(raw_tokens):
            if (token in self.vocab_list):
                tokenized[i] = self.word_to_index[token]
            else:
                tokenized[i] = self.index_UKN
        if (EOS):
            tokenized[len(raw_tokens)] = self.index_EOS
        if (SOS):
            tokenized = [self.index_SOS] + tokenized[:-1]
        return tokenized

    def embedding(self, vector_repr):
        embeddings = [self.index_to_vector[i] for i in vector_repr]
        return embeddings

    def maskings(self, index_vector_x, index_vector_y):
        '''input data type is assumped to be a list of list'''
        batch_size = len(index_vector_x)
        max_sequence_len = len(index_vector_x[0])
        decoder_lookahead = torch.ones(batch_size, max_sequence_len, max_sequence_len)
        decoder_lookahead = torch.triu(decoder_lookahead, diagonal=1)
        encoder_padding = torch.zeros(batch_size, max_sequence_len, max_sequence_len)
        decoder_padding = torch.zeros(batch_size, max_sequence_len, max_sequence_len)
        for sample in range(batch_size):
            encoder_padding_start = index_vector_x[sample].index(self.index_PAD)
            decoder_padding_start = index_vector_y[sample].index(self.index_PAD)
            for i in range(encoder_padding_start, max_sequence_len):
                encoder_padding[sample][i, :] = 1
                encoder_padding[sample][:, i] = 1
            for i in range(decoder_padding_start, max_sequence_len):
                decoder_padding[sample][i, :] = 1
                decoder_padding[sample][:, i] = 1
        encoder_self_attention = encoder_padding
        decoder_masked_self_attention = decoder_padding + decoder_lookahead
        decoder_self_attention = decoder_padding
        return encoder_self_attention, decoder_masked_self_attention, decoder_self_attention

    def response_to(self, query):
        enc_input_tokenized = self.tokenize(query, SOS = True, EOS = True)
        enc_input = self.embedding(enc_input_tokenized)
        self.transformer.eval()
        dec_input_sentence = ""
        for i in range(self.MAX_SEQ_LEN):
            dec_input_tokenized = self.tokenize(dec_input_sentence, SOS=True)
            dec_input = self.embedding(dec_input_tokenized)
            encoder_padding_mask, decoder_lookahead_mask, decoder_padding_mask = self.maskings(enc_input, dec_input)
            predictions = self.transformer(enc_input, dec_input, encoder_padding_mask, decoder_lookahead_mask, decoder_padding_mask)
            next_word = self.vocab_list[torch.argmax(predictions[0][i]).item()]
            dec_input_sentence = dec_input_sentence + next_word
            if (next_word == '<EOS>'):
                break
        return dec_input_sentence


