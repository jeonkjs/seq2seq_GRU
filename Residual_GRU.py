import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class Decoder(nn.Module):
  def __init__(self, config, embedding):
    super(Decoder, self).__init__()
    self.embedding_size = config["embedding_size"]
    self.hidden_size = config["hidden_size"]
    self.vocab_size = config["vocab_size"]
    self.embedding = embedding
    self.dropout = nn.Dropout(config["dropout"])
    self.gru = nn.GRU(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=1)
    self.linear_1 = nn.Linear(in_features=self.hidden_size*2, out_features=self.hidden_size)
    self.linear_2 = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
  
  def forward(self, input, last_hidden, encoder_outputs):
    embedded = self.embedding(input).unsqueeze(0)
    embedded = self.dropout(embedded)
    rnn_output, hidden = self.gru(embedded, last_hidden)
    rnn_output_for_attention = rnn_output.transpose(0, 1).contiguous()

    encoder_outputs = encoder_outputs.permute(1, 2, 0)
    attn_weights = rnn_output_for_attention.bmm(encoder_outputs)
    attn_weights = F.softmax(attn_weights, dim=-1)
    context = attn_weights.bmm(encoder_outputs.transpose(1, 2))
    rnn_output, context = rnn_output.squeeze(0), context.squeeze(1)
    concat_input = torch.cat(tensors=(rnn_output, context), dim=-1)
    concat_output = torch.tanh(self.linear_1(concat_input))
    output = self.linear_2(concat_output)
    return output, hidden

class Seq2Seq(nn.Module):
  def __init__(self, config):
    super(Seq2Seq, self).__init__()
    self.vocab_size = config["vocab_size"]
    self.embedding_size = config["embedding_size"]
    self.hidden_size = config["hidden_size"]
    self.max_length = config["max_length"]
    self.word_embedding = nn.Embedding(num_embeddings=self.vocab_size, 
                                       embedding_dim=self.embedding_size,
                                       padding_idx=0)

    self.bi_gru = nn.GRU(input_size=self.embedding_size,
                         hidden_size=self.hidden_size // 2,
                         num_layers=1,
                         bidirectional=True)

    self.bi_gru_2 = nn.GRU(input_size=self.embedding_size + self.hidden_size,
                        hidden_size=self.hidden_size // 2,
                        num_layers=1,
                        bidirectional=True)
                        
    self.decoder = Decoder(config, self.word_embedding)
    self.dropout = nn.Dropout(config["dropout"])
    
  def forward(self, input_features, output_features=None):
    batch_size = input_features.size()[0]
    input_feature_lengths = (input_features != 0).sum(dim=-1)
    input_features = self.word_embedding(input_features)
    input_features = input_features.transpose(0, 1)
    gru_outputs, _ = self.bi_gru(input_features)
    input_input_features = torch.cat(tensors=(gru_outputs, input_features), dim=2)
    gru_outputs_unpacked, gru_hidden_states = self.bi_gru_2(input_input_features)
    gru_hidden_states = torch.cat(tensors=(gru_hidden_states[0], gru_hidden_states[1]), dim=-1)
    gru_outputs_unpacked, gru_hidden_states = self.dropout(gru_outputs_unpacked), self.dropout(gru_hidden_states)
    decoder_input = torch.ones(size=(batch_size, ), dtype=torch.long).cuda()
    decoder_hidden = gru_hidden_states.unsqueeze(0)
    decoder_outputs = []
    if(output_features is not None):
      for step in range(self.max_length):
        decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                      decoder_hidden,
                                                      gru_outputs_unpacked)

        decoder_input = output_features[:, step]
        decoder_outputs.append(decoder_output)
          
      decoder_outputs = torch.stack(tensors=decoder_outputs, dim=0)
      decoder_outputs = decoder_outputs.transpose(0, 1)
      loss_fct = nn.CrossEntropyLoss()

      decoder_outputs = decoder_outputs.reshape(shape=(-1, self.vocab_size))
      output_features = output_features.flatten()
      loss = loss_fct(decoder_outputs, output_features)
      return loss
    else:
      for t in range(self.max_length):
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, gru_outputs_unpacked)
        decoder_input = decoder_output.argmax(dim=-1)
        decoder_outputs.append(decoder_output.argmax(dim=-1))      
      decoder_outputs = torch.stack(tensors=decoder_outputs, dim=0)
      decoder_outputs = decoder_outputs.transpose(0, 1)
      return decoder_outputs


import torch
import numpy as np

def read_data(file_path):
  with open(file_path, "r", encoding="utf8") as inFile:
    lines = inFile.readlines()
  datas = []
  for line in tqdm(lines, desc="read_data"):
    pieces = line.strip().split("\t")
    assert len(pieces) == 2
    question, answer = pieces[0].split(), pieces[1].split()
    datas.append((question, answer))
  return datas

def read_vocab_data(vocab_data_path):
  word2idx, idx2word = {}, {}

  with open(vocab_data_path, "r", encoding="utf8") as inFile:
    lines = inFile.readlines()

  for line in lines:
    word = line.strip()
    word2idx[word] = len(word2idx)
    idx2word[word2idx[word]] = word
  return word2idx, idx2word

def convert_data2feature(datas, max_length, word2idx):
  input_features, output_features = [], []
  for input_sequence, output_sequence in tqdm(datas, desc="convert_data2feature"):
    input_feature, output_feature = np.zeros(shape=(max_length), dtype=np.int), np.zeros(shape=(max_length), dtype=np.int)
  
    for index in range(len(input_sequence[:max_length])):
      input_feature[index] = word2idx[input_sequence[index]]
    for index in range(len(output_sequence[:max_length])):
      output_feature[index] = word2idx[output_sequence[index]]
    
    output_feature[index+1] = word2idx["--END--"]
    
    input_features.append(input_feature)
    output_features.append(output_feature)
  
  input_features = torch.tensor(input_features, dtype=torch.long)
  output_features = torch.tensor(output_features, dtype=torch.long)
  return input_features, output_features

import os
import re
import numpy as np
import torch
from torch.utils.data import (DataLoader, TensorDataset)
import torch.optim as optim

def train(config):
  train_datas = read_data(config["train_data_path"])
  word2idx, idx2word = read_vocab_data(config["vocab_data_path"])
  train_input_features, train_output_features \
                = convert_data2feature(train_datas, config["max_length"], word2idx)

  train_features = TensorDataset(train_input_features, train_output_features)
  train_dataloader = DataLoader(train_features, shuffle=True, batch_size=config["batch_size"])

  model = Seq2Seq(config).cuda()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  for epoch in range(config["epoch"]):
    model.train()
    losses = []
    for step, batch in enumerate(tqdm(train_dataloader, desc="epoch_{}_train".format(epoch + 1))):
      batch = tuple(t.cuda() for t in batch)
      input_features, output_features = batch[0], batch[1]
      loss = model(input_features, output_features)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      losses.append(loss.data.item())
    torch.save(model.state_dict(), os.path.join(output_dir, "epoch_{}.pt".format(epoch + 1)))
    print("Average loss : {}\n".format(np.mean(losses)))

import os
from torch.utils.data import (DataLoader, TensorDataset)

def test(config):
# 평가 데이터 읽기
  test_datas = read_data(config["test_data_path"])
# 어휘 딕셔너리 생성
  word2idx, idx2word = read_vocab_data(config["vocab_data_path"])
  test_input_features, test_output_features \
                = convert_data2feature(test_datas, config["max_length"], word2idx)
# 평가 데이터를 batch 단위로 추출하기 위한 DataLoader 객체 생성
  test_features = TensorDataset(test_input_features, test_output_features)
  test_dataloader = DataLoader(test_features, shuffle=False, batch_size=1)
# Seq2Seq 모델 객체 생성
  model = Seq2Seq(config).cuda()
# 사전학습한 모델 파일로부터 가중치 불러옴
  model.load_state_dict(torch.load(os.path.join(config["output_dir_path"], config["trained_model_name"])))
  model.eval()

  for step, batch in enumerate(tqdm(test_dataloader, desc="test")):
    batch = tuple(t.cuda() for t in batch)
# 음절 데이터, 각 데이터의 실제 길이, 라벨 데이터
    input_features, output_features = batch[0], batch[1]
# 모델 평가
    predicts = model(input_features)
    predicts, input_features = predicts[0], input_features[0]
# Tensor 를 리스트로 변경
    predicts = predicts.cpu().numpy().tolist()
    input_features = input_features.cpu().numpy().tolist()
      if (step < 5):
        print()
        print("input_features : ", end=" ")
      for idx in input_features:
        word = idx2word[idx]
      if(word == "--PAD--"):
        break
      print(word, end=" ")
    print()
    
    print("predicts : ", end=" ")
    for idx in predicts:
      word = idx2word[idx]
      if(word == "--END--"):
        break
      print(word, end=" ")
    print()

if(__name__=="__main__"):
  root_dir = "root_dir"
  output_dir = os.path.join(root_dir, "output")
  
  if(not os.path.exists(output_dir)):
    os.makedirs(output_dir)
  config = {"mode": "test",
            "trained_model_name": "epoch_{}.pt".format(300),
            "train_data_path": os.path.join(root_dir, "train.txt"),
            "test_data_path": os.path.join(root_dir, "test.txt"),
            "output_dir_path": output_dir,
            "vocab_data_path": os.path.join(root_dir, "vocab.txt"),
            "vocab_size": 19353,
            "embedding_size": 100,
            "hidden_size": 200,
            "max_length": 52,
            "epoch": 300,
            "batch_size": 32,
            "dropout": 0.3
            }

  if (config["mode"] == "train"):
    train(config)
  else:
    test(config)
