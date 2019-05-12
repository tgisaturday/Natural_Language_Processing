import pickle
import numpy as np
import torch

f_embed = open("embeddings/CBOW/embedding.pkl","rb")
f_dict = open("embeddings/CBOW/word_to_ix.pkl","rb")

embeddings = pickle.load(f_embed)
word_to_ix = pickle.load(f_dict)
ix_to_word =  {y:x for x,y in word_to_ix.items()}

f_test = open("data/questions_words.txt","r")

total = 0
count = 0
unk_idx = word_to_ix["<UNK>"]
for aLine in f_test:
    if aLine.strip()[0]!= ":" :
        raw = aLine.strip().split()
        
        first_idx = word_to_ix.get(raw[0].lower())
        if first_idx == None:
            first_idx = unk_idx
        first = embeddings[first_idx]
        
        second_idx = word_to_ix.get(raw[1].lower())
        if second_idx == None:
            second_idx = unk_idx
        second = embeddings[second_idx]

        third_idx = word_to_ix.get(raw[2].lower())
        if third_idx == None:
            third_idx = unk_idx
        third = embeddings[third_idx]
        
        target = first - second + third

        
        dist_min = torch.norm(embeddings[0] - target,2)
        dist_idx = 0

        for i in range(1,len(embeddings)):
            dist = torch.norm(embeddings[i] - target,2)
            if dist < dist_min:
                dist_min = dist
                dist_idx = i
        if ix_to_word[dist_idx] == raw[3].lower():
            count +=1
        total += 1

print("Word Analogy Task Accuracy: {0:.3f}".format(count/total))
        
