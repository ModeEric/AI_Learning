import torch.nn as nn
import torch
import torch.nn.functional as F
#Work in progress, need to finish decoder and add batching (partially done)

def dot_attention(Q,K,V,d):
    return nn.Softmax((Q @ K.transpose())/d) @ V

class SelfDecoderBlock(nn.Module):
    def __init__(self,input_length, embedding_size, d_model, heads=2):
        self.input_length = input_length
        self.embedding_size = embedding_size
        self.d_model = d_model
        self.heads = heads
        self.projectionk = []
        self.projectionq = []
        self.projectionv = []
        self.projectiono = []
        for i in range(heads):
            self.projectionk.append(nn.Parameter(torch.zeros(embedding_size,d_model)))
            self.projectionq.append(nn.Parameter(torch.zeros(embedding_size,d_model)))
            self.projectionv.append(nn.Parameter(torch.zeros(embedding_size,d_model)))
            self.projectiono.append(nn.Parameter(torch.zeros(heads*d_model,embedding_size)))
    def forward(self,input):
        results = dot_attention(input@self.projectionq[i], input@self.projectionk[i], input@self.projectionv[i])
        for i in range(1, self.heads):
            results = torch.concat(results,dot_attention(input@self.projectionq[i], input@self.projectionk[i], input@self.projectionv[i]))
        results = results @ self.projectiono[i]
        return results
        
    

class SelfDecoder(nn.Module):
    def __init__(self,input_length=128,embedding_size=256, num_blocks=4, d_model = 512, heads=2, num_classes = 2):
        self.input_length = input_length
        self.embedding_size = embedding_size
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.heads = heads
        self.finallinear = nn.Parameter(num_classes,d_model)
        self.layers = [SelfDecoderBlock(input_length,embedding_size,d_model,heads) for i in range(num_blocks)]
        self.softmax = nn.Softmax(dim=1)
    def fixed_pos_embed(self,input):
        new_tensor = torch.zeros(input.shape)
        for i in range(input.shape[0]):
            for pos in range(input.shape[1]):
                if i%2==0:
                    new_tensor[i,pos] = torch.sin((pos/10000)**(2*i)/self.d_model) #Exponential is equally spaced throughout steps, perhaps can take advantage for speedup
                else:
                    new_tensor[i,pos] = torch.cos((pos/10000)**(2*i)/self.d_model)
        return new_tensor
    def forward(self,inputs):
        embeddings = self.fixed_pos_embed(inputs)
        for i in range(self.num_blocks):
            embeddings = self.layers[i](embeddings())
        output = self.finallinear @ embeddings
        return self.softmax(output, dim=1)
