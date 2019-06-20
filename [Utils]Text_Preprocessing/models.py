import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size ,embedding_dim, hidden_size, dropout_p,  num_layers=1):
        super(EncoderRNN, self).__init__()
        self.input_dim = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_size
        self.embedding = nn.Embedding(self.input_dim, self.embedding_dim)
        self.rnn = nn.GRU(self.embedding_dim, self.hidden_dim,
                          bidirectional=True,num_layers=num_layers,
                          dropout=dropout_p)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_input, hidden):
        embedded = self.embedding(encoder_input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden
    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_dim, device=device)
    
class DecoderRNN(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_size, dropout_p, max_length, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_size
        self.output_dim = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_dim,self.embedding_dim)
        self.rnn = nn.GRU(self.embedding_dim, self.output_dim, num_layers=num_layers,dropout=dropout_p)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, decoder_input, hidden):
        embedded = self.embedding(decoder_input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        output = F.relu(output)
        output, hidden = self.rnn(embedded, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_dim, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_size, dropout_p, max_length, num_layers=1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_size
        self.output_dim = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        embedded = self.embedding(decoder_input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        self.attn = nn.Linear(self.hidden_dim * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn = nn.GRU(self.embedding_dim, self.hidden_dim,
                          num_layers=num_layers,dropout=dropout_p)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, decoder_input, hidden, encoder_outputs):
        embedded = self.embedding(decoder_input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_dim, device=device)

def train_autoencoder(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,teacher_forcing_ratio, max_length, SOS_token, EOS_token, device):  
    encoder_hidden = encoder.initHidden(device)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_dim, device=device)
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, hidden, cell)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
                
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length        