import json
import sys
import os
import codecs

def converter(filename):
    rawdata = json.loads(open(filename,'rt',encoding='utf-8-sig').read())
    newfile = '../data/NER_'+filename.split('_')[-1].split('.')[0]+'.txt'
    wfp = open(newfile,'w')
    sentence_id = len(rawdata['sentence'])
    for i in range(sentence_id):
        sentence = rawdata['sentence'][i]
        morp_id = len(sentence['morp'])
        word_len = len(sentence['word'])
        NE_len = len(sentence['NE'])
        cur_word_idx = 0
        cur_NE_idx = 0
        if NE_len == 0:
            for j in range(morp_id):
                cur_morp = sentence['morp'][j]  
                print(cur_morp['lemma']+'/'+cur_morp['type']+' O',file=wfp)
            print(file=wfp)
            continue
        for j in range(morp_id):
            cur_morp = sentence['morp'][j]
            if cur_word_idx+1 < word_len:
                if (cur_morp['lemma'] not in sentence['word'][cur_word_idx]['text']) and (cur_morp['lemma'] in sentence['word'][cur_word_idx+1]['text']):
                    cur_word_idx += 1
            if cur_NE_idx+1 < NE_len:
                if (cur_morp['lemma'] not in sentence['NE'][cur_NE_idx]['text']) and (cur_morp['lemma'] in sentence['NE'][cur_NE_idx+1]['text']):
                    cur_NE_idx += 1
                
            cur_word = sentence['word'][cur_word_idx]
            cur_NE = sentence['NE'][cur_NE_idx]    
            if cur_word['text'] == cur_NE['text']:
                if cur_morp['lemma'][0] == cur_word['text'][0]:
                    bio = 'B'
                else:
                    bio = 'I'
                tag = bio+'-'+cur_NE['type']
            else:
                tag = 'O'
            print(cur_morp['lemma']+'/'+cur_morp['type']+' '+tag, file=wfp)
        print(file=wfp)

            
              
        
        

if __name__=='__main__':
    converter("../data/2016lipexpo_NERcorpus_dev.json")
    converter("../data/2016lipexpo_NERcorpus_test.json")
    converter("../data/2016lipexpo_NERcorpus_train.json")
