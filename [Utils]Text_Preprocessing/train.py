import os
import sys
import json
import time
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import utils.data_helper as data_helper
import utils.utils as utils
import models



def train(dataset_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    #Step 0: load sentences, labels, and training parameters
    dataset = 'data/'+dataset_name+'_csv/train.csv'
    testset = 'data/'+dataset_name+'_csv/test.csv'
    parameter_file = "./parameters_RNN.json"
    params = json.loads(open(parameter_file).read())
    learning_rate = params['learning_rate']
    embedding_dim = params["embedding_dim"]
    dropout_p = params["dropout_prob"]
    hidden_size = params["rnn_size"]
    batch_size = params["batch_size"]
    num_epochs = params["num_epochs"]
    teacher_forcing_ratio = params["teacher_forcing_ratio"]
    min_freq = params["min_freq"]
    cut_length = params["max_length"]
    if params["enable_attention"] == 1:
        enable_attention = True
    else:
        enable_attention = False
    
    x_raw, y_raw, df, labels = data_helper.load_data_and_labels(dataset,dataset_name, cut_length,True)
    x_test_raw, y_test_raw, df_test, labels_test = data_helper.load_data_and_labels(testset,dataset_name, cut_length,False)
    
    vocabulary = utils.Vocab(dataset_name, min_freq)
    vocabulary.add_words(x_raw)   

    #Step 1: pad each sentence to the same length and map each word to an id
    max_length = max([len(x.split(' ')) for x in x_raw])
    print('The maximum length of all sentences: {}'.format(max_length))    
    

    SOS_token = vocabulary.word2idx["SOS"]
    EOS_token = vocabulary.word2idx["EOS"]
    
    usage_ratio = round(len(vocabulary.word2idx) / vocabulary.n_words,4)*100

    print("Total number of words: {}".format(vocabulary.n_words))
    print("Number of words we will use: {}".format(len(vocabulary.word2idx)))
    print("Percent of words we will use: {0:.2f}%".format(usage_ratio))    

    # Apply convert2idx to clean_summaries and clean_texts
    idx_texts, word_count, unk_count = vocabulary.convert2idx(x_raw, eos=True)
    idx_test_texts, _ , _ = vocabulary.convert2idx(x_test_raw, eos=True)    
    unk_percent = round(unk_count/word_count,4)*100

    print("Total number of UNKs in  texts: {}".format(unk_count))
    print("Percent of words that are UNK: {0:.2f}%".format(unk_percent))
    
    #Step 2: pad each sentence to the same length and map each word to an id
    x_idx = vocabulary.pad_sentence_batch(idx_texts)
    x_test_idx = vocabulary.pad_sentence_batch(idx_test_texts)
    x = np.array(x_idx)
    y = np.array(y_raw)
    x_test = np.array(x_test_idx)
    y_test = np.array(y_test_raw)
      
    #t = np.array(list(len(x) for x in x_idx))
    #t_test = np.array(list(len(x) for x in x_test_idx))

    #Step 3: shuffle the train set and split the train set into train and dev sets
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    #t_shuffled = t[shuffle_indices]
    x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, test_size=0.1)
    
    #Step 4: save the labels into labels.json since predict.py needs it
    if not os.path.exists('results/{}/'.format(dataset_name)):
        os.makedirs('results/{}/'.format(dataset_name))
    if not os.path.exists('results/{}/models/'.format(dataset_name)):
        os.makedirs('results/{}/models/'.format(dataset_name))
    with open('results/{}/labels.json'.format(dataset_name), 'w') as outfile:
        json.dump(labels, outfile, indent=4)

    print('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
    print('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))
    #print('t_train: {}, t_dev: {}, t_test: {}'.format(len(t_train), len(t_dev), len(t_test)))

    #Step 4: build model
    vocab_size = len(vocabulary.word2idx)
    encoder = models.EncoderRNN(vocab_size,embedding_dim, 
                                hidden_size,dropout_p).to(device)
    if enable_attention == True:
        decoder = models.AttnDecoderRNN(vocab_size, embedding_dim, 
                                      hidden_size, dropout_p, max_length).to(device)
    else:
        decoder = models.DecoderRNN(vocab_size, embedding_dim, 
                                    hidden_size, dropout_p, max_length).to(device)
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()
    
    #Use input as a target (Autoencoder)
    train_input = torch.from_numpy(np.asarray(x_train)).long()
    train_target = torch.from_numpy(np.asarray(x_train)).long()
    dataset_train = data_utils.TensorDataset(train_input, train_target)
    train_loader =  data_utils.DataLoader(dataset_train, batch_size, shuffle=True, 
                                     num_workers=4, pin_memory=False)


    for epoch in range(num_epochs):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
        
        for i, (context, target) in enumerate(train_loader):
            context = context.to(device)
            target = target.to(device)
            loss = models.train_autoencoder(context, target, encoder, decoder,
                          encoder_optimizer, decoder_optimizer, criterion,
                          teacher_forcing_ratio, max_length, SOS_token, EOS_token, device)
            
            print_loss_total += loss
            plot_loss_total += loss

            if i % 1000 == 0:
                print_loss_avg = print_loss_total / 1000
                print_loss_total = 0
                print("loss: %.4f, steps: %dk" % (loss), ((i+1)/1000))

            #if i % 1000 == 0:
                #plot_loss_avg = plot_loss_total / 1000
                #plot_losses.append(plot_loss_avg)
                #plot_loss_total = 0            
            
        #showPlot(plot_losses)
        end = time.time()
        print("epochs: %d" % (epoch+1))
        print("time eplased: %d seconds" % (end-start))
        print("mean loss: %.4f" % (total_loss / (train_input.shape[0] // BATCH_SIZE)))
        torch.save(model.state_dict(), "results/{}/models/epoch{}.model".format(dataset_name,epoch))

if __name__ == '__main__':
    train(sys.argv[1])