import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.lstm = nn.LSTM(input_size = embed_size,hidden_size = hidden_size,batch_first = True)
        self.fc = nn.Linear(hidden_size,vocab_size)
        self.s = nn.Softmax(dim=1)
    
    def forward(self, features, captions):
        batch_size = features.size(0)
        hidden_state = torch.zeros((1,batch_size, self.hidden_size)).cuda()
        cell_state = torch.zeros((1,batch_size, self.hidden_size)).cuda()
        outputs = torch.empty((batch_size, captions.shape[1], self.vocab_size)).cuda()
        output = torch.empty((batch_size, 1, self.vocab_size)).cuda()
        embeddings = self.word_embeddings(captions)
        input = torch.cat((features.unsqueeze(1),embeddings[:,:-1,:]),1)   #excluded the last word using embeddings[:,:-1.:] as per previous review
        hidden_state,cell_state = self.lstm(input,(hidden_state,cell_state))
        output = self.fc(hidden_state)
        return output
        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sampled_ids = []
        inputs = inputs.unsqueeze(1)
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)          
            outputs = self.fc(hiddens.squeeze(1))  
            outputs = self.s(outputs)
            _,predicted = outputs.max(1)
            sampled_ids.append(predicted.item())
            inputs = self.word_embeddings(predicted)                       
            inputs = inputs.unsqueeze(1)                        
        #sampled_ids = torch.stack(sampled_ids, 1)              
        return list(sampled_ids)