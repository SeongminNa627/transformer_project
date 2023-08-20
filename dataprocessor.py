

import os
import ujson
import json
import csv
import gensim
import nltk

# path = os.getcwd()
# corpus_name = "preprocessed.json"
# file = os.path.join(path, corpus_name)
# with open(file, "r") as datafile:
#     data = ujson.load(datafile)
# fieldnames = data[0].keys()
#
# # with open("preprocessed.csv", "w") as writefile:
# #     writer = csv.DictWriter(writefile, fieldnames = fieldnames)
# #     writer.writeheader()
# #     for datum in data:
# #         writer.writerow(datum)
# path = os.getcwd()
# corpus_name = "preprocessed.json"
# file = os.path.join(path, corpus_name)
# with open(file, "r") as datafile:
#     data = ujson.load(datafile)
# fieldnames = data[0].keys()
#
# # with open("preprocessed.csv", "w") as writefile:
# #     writer = csv.DictWriter(writefile, fieldnames = fieldnames)
# #     writer.writeheader()
# #     for datum in data:
# #         writer.writerow(datum)
#
# collected = {}
# temp_conversation_id = data[0]['conversation_id']
# temp_list = []
# for utterance in data:
#
#     if (utterance['conversation_id'] == temp_conversation_id):
#         del utterance['conversation_id']
#         temp_list.append(utterance)
#     else:
#         collected.update({temp_conversation_id: temp_list})
#         temp_conversation_id = utterance['conversation_id']
#         del utterance['conversation_id']
#         temp_list = [utterance]
# for key, value in collected.items():
#     if (len(value)%2 ==1):
#         new_id = 'L' + str(int(value[0]['id'][1:]) + 1)
#         new_list = [{'id': new_id, 'text': "...", 'reply-to': value[0]['id'], 'parsed': ['...']}]
#         value = new_list + value
#         collected.update({key: value})
#
# input_sentence = []
# output_sentence = []
#
#
# for key, value in collected.items():
#     value.reverse()
#     for i in range(len(value)):
#         if(i%2 == 0):
#             input_sentence.append(str(value[i]['text']))
#         else:
#             output_sentence.append(str(value[i]['text']))
#
# sentences_input = [nltk.word_tokenize(sentence) for sentence in input_sentence]
# sentences_output = [nltk.word_tokenize(sentence) for sentence in output_sentence]
# with open("input_file.json", "w") as inputf:
#     json.dump(input_sentence, inputf)
# with open("output_file.json", "w") as outputf:
#     json.dump(output_sentence, outputf)
#




with open("input_file.json", "r") as inputfile:
    input_sentences = json.load(inputfile)
with open("output_file.json", "r") as outputfile:
    output_sentences = json.load(outputfile)

sentences = [nltk.word_tokenize(sentence) for sentence in input_sentences + output_sentences]
model = gensim.models.Word2Vec(sentences, min_count = 3,vector_size= 300)
vocab_list = model.wv.index_to_key
print("TOTAL Sentences: ", len(input_sentences + output_sentences))
valid_input_sentences = []
valid_output_sentences = []

i = 0
while i < (len(input_sentences)):
    input_tokens = nltk.word_tokenize(input_sentences[i])
    output_tokens = nltk.word_tokenize(output_sentences[i])
    tokens = input_tokens + output_tokens
    validity = [token in vocab_list for token in tokens]
    if all(validity):
        valid_input_sentences.append(input_sentences[i])
        valid_output_sentences.append(output_sentences[i])
    i = i + 1


print("Valid Sentences: ", len(input_sentences + output_sentences))
print("input_sentences: ", len(input_sentences))
print("output_sentences: ", len(output_sentences))

with open("valid_input_file.json", "w") as inputfile:
    json.dump(valid_input_sentences, inputfile)
with open("valid_output_file.json", "w") as outputfile:
    json.dump(valid_output_sentences, outputfile)



