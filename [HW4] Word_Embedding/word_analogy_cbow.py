import pickle
import numpy as np

f_embed = open("embeddings/CBOW/embedding.pkl","rb")
f_dict = open("embeddings/CBOW/word_to_ix.pkl","rb")

embeddings = pickle.load(f_embed)
word_to_ix = pickle.load(f_dict)
ix_to_word =  {y:x for x,y in word_to_ix.items()}

f_test = open("data/questions_words.txt","r")

total = 0
count = 0

for aLine in f_test:
    if aLine.strip()[0]!= ":" :
        raw = aLine.strip().split()
        try: 
            first = embeddings[word_to_ix[raw[0].lower()]]
        except:
            first = embeddings[word_to_ix["<UNK>"]]

        try: 
            second = embeddings[word_to_ix[raw[1].lower()]]
        except:
            second = embeddings[word_to_ix["<UNK>"]]

        try: 
            third = embeddings[word_to_ix[raw[2].lower()]]
        except:
            third = embeddings[word_to_ix["<UNK>"]]
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
        
