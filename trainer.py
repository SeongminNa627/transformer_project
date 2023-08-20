
import json
import nltk
import gensim
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import Transformer as TF

with open("input_file.json", "r") as inputfile:
    input_sentences = json.load(inputfile)
with open("output_file.json", "r") as outputfile:
    output_sentences = json.load(outputfile)



valid_input_sentences = []
valid_output_sentences = []

i = 0
while i < (len(input_sentences)):
    input_tokens = nltk.word_tokenize(input_sentences[i])
    output_tokens = nltk.word_tokenize(output_sentences[i])
    if len(input_tokens) < 98 and len(output_tokens) < 98:
        valid_input_sentences.append(input_sentences[i])
        valid_output_sentences.append(input_sentences[i])
    i = i + 1

D_MODEL = 304
sentences = [nltk.word_tokenize(sentence) for sentence in valid_input_sentences + valid_output_sentences]
model = gensim.models.Word2Vec(sentences, min_count = 3,vector_size= D_MODEL)
vocab_list = model.wv.index_to_key #<class 'list'>  vocab_list[3] = 'you'
word_to_index = model.wv.key_to_index #<class 'dict'> word_to_index['you'] = 3
#model.wv['you'] == model.wv[3] both return a vector representation of 'you'
#len(model.wv) = 28714

index_to_vector = {i : model.wv[j] for i, j in enumerate(range(len(model.wv)))}
MAX_SEQ_LEN = 100
VOCAB_SIZE = len(model.wv)

embed_SOS = torch.randn(1, D_MODEL).squeeze()
index_SOS = VOCAB_SIZE
index_to_vector[index_SOS] = embed_SOS
vocab_list.append('<SOS>')
VOCAB_SIZE = VOCAB_SIZE + 1

embed_EOS = torch.randn(1, D_MODEL).squeeze()
index_EOS = index_SOS + 1
index_to_vector[index_EOS] = embed_EOS
vocab_list.append('<EOS>')
VOCAB_SIZE = VOCAB_SIZE + 1


embed_UKN = torch.randn(1, D_MODEL).squeeze()
index_UKN = index_EOS + 1
index_to_vector[index_UKN] = embed_UKN
vocab_list.append('<UKN>')
VOCAB_SIZE = VOCAB_SIZE + 1


embed_PAD = torch.randn(1, D_MODEL).squeeze()
index_PAD = index_UKN + 1
index_to_vector[index_PAD] = embed_PAD
vocab_list.append('<PAD>')
VOCAB_SIZE = VOCAB_SIZE + 1


def tokenize(sentence, SOS = False, EOS = False):
    '''Given a sentence, return a vector representation (index) of the sentence'''
    tokenized = [index_PAD for _ in range(MAX_SEQ_LEN)]
    raw_tokens = nltk.word_tokenize(sentence)
    for i, token in enumerate(raw_tokens):
        if (token in vocab_list):
            tokenized[i] = word_to_index[token]
        else:
            tokenized[i] = index_UKN
    if (EOS):
        tokenized[len(raw_tokens)] = index_EOS
    if (SOS):
        tokenized = [index_SOS] + tokenized[:-1]
    return tokenized

def embedding(vector_repr):
    embeddings = torch.tensor([list(index_to_vector[i]) for i in vector_repr])
    return embeddings



class DataSet(Dataset):
    def __init__(self, valid_input_sentences, valid_output_sentences):

        self.input = valid_input_sentences

        self.output = valid_output_sentences
    def __len__(self):
        return len(self.input)
    def __getitem__(self, index):
        return self.input[index], self.output[index]

def maskings(index_vector_x, index_vector_y, heads):
    '''input data type is assumped to be a list of list'''
    batch_size = len(index_vector_x)
    max_sequence_len = len(index_vector_x[0])
    decoder_lookahead= torch.ones(batch_size, heads, max_sequence_len, max_sequence_len)
    decoder_lookahead = torch.triu(decoder_lookahead, diagonal = 1)
    encoder_padding = torch.zeros(batch_size, heads, max_sequence_len, max_sequence_len)
    decoder_padding= torch.zeros(batch_size, heads, max_sequence_len, max_sequence_len)
    for sample in range(batch_size):
        for h in range(heads):
            encoder_padding_start = index_vector_x[sample].index(index_PAD)
            decoder_padding_start = index_vector_y[sample].index(index_PAD)
            for i in range(encoder_padding_start, max_sequence_len):
                encoder_padding[sample][h][i,: ] = 1
                encoder_padding[sample][h][ :,i] = 1
            for i in range(decoder_padding_start, max_sequence_len):
                decoder_padding[sample][h][i,: ] = 1
                decoder_padding[sample][h][:, i] = 1
    encoder_self_attention = encoder_padding
    decoder_masked_self_attention = decoder_padding + decoder_lookahead
    decoder_self_attention = decoder_padding
    return encoder_self_attention, decoder_masked_self_attention, decoder_self_attention
def label_probabilities(labels, max_seq_len, vocab_size):
    def label_to_probability(sample, max_seq_len, vocab_size):
        label_to_prob = torch.zeros(max_seq_len, vocab_size)
        for i, word_index in enumerate(sample):
            label_to_prob[i][word_index] = 1
        return label_to_prob
    return torch.tensor([label_to_probability(label, max_seq_len, vocab_size) for label in labels])


EPOCHS = 30
BATCH_SIZE = 30
NUM_HEADS = 8
NUM_LAYERS = 2

ds = DataSet(valid_input_sentences, valid_output_sentences)

transformer = TF.Transformer(D_MODEL, NUM_HEADS, NUM_LAYERS, VOCAB_SIZE)
loss_fn = nn.CrossEntropyLoss(ignore_index=index_PAD)
optimizer = optim.Adam(transformer.parameters(), lr = 0.0001)


for j, epoch in enumerate(range(EPOCHS)):
    train_dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last = True)
    print("EPOCH ", j)
    for iteration, (samples, labels) in enumerate(train_dataloader):
        enc_input = [tokenize(sample, SOS = True, EOS = True) for sample in samples]
        dec_input = [tokenize(label, SOS = True) for label in labels]
        x = torch.stack([embedding(input_sample) for input_sample in enc_input])
        y = torch.stack([embedding(input_label) for input_label in dec_input])
        y_labels = torch.tensor([tokenize(label, EOS = True) for label in labels])
        # TODO: Create the masking for the samples and the labels
        encoder_padding_mask, decoder_lookahead_mask, decoder_padding_mask = maskings(enc_input, dec_input, NUM_HEADS)
        # TODO: Pass the samples, maskings into the Transformer
        optimizer.zero_grad()
        predictions = transformer(x, y, encoder_padding_mask, decoder_lookahead_mask, decoder_padding_mask)
        # TODO: Calculate the loss based on the labels
        loss = loss_fn(predictions.view(-1, VOCAB_SIZE), y_labels.view(-1))
        # TODO: Update the gradient
        loss.backward()
        optimizer.step()
        # TODO: Print out a sample and the loss (iteration %100 == 0)
        if (iteration % 100 == 0):
            print(f"Iteration {iteration} : {loss.item()}")
    test_input_tokenized = tokenize("Hi! How are you?", SOS=True, EOS=True)
    enc_input = torch.stack([embedding(test_input_tokenized)])
    transformer.eval()
    dec_input_sentence = ""
    for i in range(10):
        dec_input_tokenized = tokenize(dec_input_sentence, SOS=True)
        dec_input = torch.stack([embedding(dec_input_tokenized)])
        encoder_padding_mask, decoder_lookahead_mask, decoder_padding_mask = maskings( [test_input_tokenized], [dec_input_tokenized], NUM_HEADS)
        predictions = transformer(enc_input, dec_input, encoder_padding_mask, decoder_lookahead_mask,
                                       decoder_padding_mask)
        next_word = vocab_list[torch.argmax(predictions[0][i]).item()]
        dec_input_sentence = dec_input_sentence + " " +  next_word
        if (next_word == '<EOS>'):
            break
    print("Prediction of a response to \"Hi! How are you?\": ", dec_input_sentence, " ... ")
    print("-------------------------------------------------------------------------------")
transformer_file_name = "saved_transformer.pth"
torch.save(transformer.state_dict(), transformer_file_name)
optimizer_file_name = "saved_optimizer.pth"
torch.save(optimizer.state_dict(), optimizer_file_name)












