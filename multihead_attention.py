import torch
import torch.nn as nn
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.head_dim = d_model // self.num_head
        self.W_qkv = nn.Linear(d_model, d_model * 3)
        self.linear = nn.Linear(self.d_model, self.d_model)

    def forward(self, x, padding_mask = None):
        '''
        :param x: tensor with size(batch_size, seq_len,embedding_size)
                    x = [
                         START:   [some values]
                         You:     [.1, -.04, .3, 3]
                         Are:     [1, -1, 0, .3]
                         Welcome: [.1, -.1, -.3, .3]
                         PAD:     [some values]
                         PAD:     [some values]
                         EOS:     [some values]
                        ]

        1. By going through self.W_qkv.forward(x) we get a tensor horizontally concatenated along embedding_size, each of which is Q, K, V.
            1 - 1. Assuming that the first dimension is going to be the batch dimension, we are splitting it into the batch_num tensors first.
        2. Split them into the number of heads and squeeze them to get rid of the dummy dimension created as a result of split.
        3. For each head of each example, we split the head into into a tuple of Q,K,V.
            3 - 1. For each example in this tuple, we will split the example into subtensor according to the number of heads.
                    - The resulting structure would be [(tensor1, tensor2,...,h_tensor), (tensor1,tensor2,...,h_tensor),....,(tensor1, tensor2,...,h_tensor)_m]
                    where h = number of heads, and m is the number of example (batch_dim)
            3 - 2. For each example, we have heads_Q, heads_K, heads_V prepared. For each head, we have Q_list, K_list, V_list prepared and perform hsplit on the head into 3 which results in q, k, v for one head.
                  /Then, we do this for every head and store all the qs into Q_list, ks into K_list, and vs into V_list.
                  /We then, stack them to form the Q, K and V for one example and put it in the example_Q.
                  /Then we stack the example Q, K, V to form the final Q,K,V with size being (batch_size, number of heads, and seq_len) for each.
        4. Apply QK^T  / sqrt(head_dim = embedding/heads)
        5. Feed this into another layer, which is a dropout layer with p = 0.4
        6. Multiply the Resulting tensor by Value tensor
        7. Now we reshape the resulting tensor into one with size being (batch_size, seq_len, embedding_size)
            which is our attention score.

        '''
        batch_size, seq_len, d_model = x.size()
        mixed_qkv = self.W_qkv.forward(x)
        mixed_qkv = torch.vsplit(mixed_qkv, batch_size)
        '''
        (tensor(), tensor(), ...,tensor())
        each tensor is the example in the batch.
        '''

        mixed_qkv = [torch.hsplit(torch.squeeze(example), self.num_head) for example in mixed_qkv]
        '''
        [(tensor11(), tensor12(), ... ,tensor1h()), (tensor21(), tensor22(), ... , tensor2h()),...., (tensorb1(), tensorb2(), ... , tensorbh())]
        each tuple is the example in the batch.
        each tensor in the example is the head.
        '''
        example_Q, example_K, example_V= [], [], []
        for example in mixed_qkv:
            Q_list = []  # tensor11q, tensor12q, tensor13q,...,tensor1hq, tensor21q, tensor22q,...,tensorb1q,tensorb2q...tensorbhq.
            K_list = []  # tensor11k, tensor12k, tensor13k,...,tensor1hk, tensor21k, tensor22k,...,tensorb1k,tensorb2q...tensorbhk.
            V_list = []  # tensor11v, tensor12v, tensor13v,...,tensor1hv, tensor21v, tensor22v,...,tensorb1v,tensorb2v...tensorbhv.
            for head in example:
                # divide each head into 3 subtensor, Q, K, V.
                q, k, v = torch.hsplit(head, 3)
                Q_list.append(q), K_list.append(k), V_list.append(v)
            example_Q.append(torch.stack(Q_list)), example_K.append(torch.stack(K_list)), example_V.append(torch.stack(V_list))
        Q = torch.stack(example_Q)  # (batch_size, num_head, seq_len, head_dim/3 = q,k,v)
        '''
        tensor([[[[-0.7890,  0.5960],
                  [-0.3010, -1.4791],
                  [-0.2249, -0.1266]],

                 [[-0.2180, -0.0667],
                  [ 0.3718, -0.4771],
                  [-0.0250, -0.8203]]],


                [[[ 0.5109,  0.5478],
                  [-0.5112, -0.1711],
                  [-0.1109,  0.6513]],

                 [[ 0.1004, -0.1657],
                  [-0.0488,  0.2856],
                  [-0.2192, -0.9546]]]])
        '''
        K = torch.stack(example_K)
        V = torch.stack(example_V)


        atten_score = torch.matmul(Q, K.transpose(-1, -2))

        masking = float('-10e9')* padding_mask

        atten_score = atten_score + masking
        atten_score = atten_score / np.sqrt(self.head_dim)

        F = torch.nn.Softmax(-1)

        atten_score = F(atten_score)
        out = torch.matmul(atten_score, V)
        values = torch.reshape(out, (batch_size, seq_len, d_model))
        values = self.linear(values)
        return values
    # def scaled_dot_product_attention(self, q, k, v):
    #     s = torch.matmul(q, k.transpose(-2, -1))/np.sqrt(self.head_dim)
    #     s = functorch.dim.softmax(s,-1)
    #     attention_score = torch.matmul(s, v)
    #     return attention_score

# def main():
#     test = torch.randn(1, 50, 304)
#     model = MultiHeadAttention(1, 50, 304, 8)
#     out = model(test)
# main()