import math

def train(train_file):
    tfp = open(train_file,'r')
    #특정 단어가 나왔을 때 그 단어가 특정 POS일 확률: observation probability
    #특정 POS에서 다음 POS로 넘어갈 확률: transition probability
    POS_dict = dict()
    trans_dict = dict()
    observ_dict = dict()
    word_dict = dict()
    trans_prob_dict = dict()
    observ_prob_dict = dict()
    train_data = tfp.readlines()
    tfp.close()
    POS_dict['<s>'] = len(train_data)
    #I. HMM 구축을 위한 train data 분석 및 사전 구축
    for aLine in train_data:
        word = aLine.strip().split()
        if len(word)==0:
            continue
        prev = '<s>'
        for token in word[1].split('+'):
            try:
                morph = token.split('/')[0]
                pos = token.split('/')[1]
            except:
                #오타 및 불규칙 형태 무시
                continue
            trans_key = pos +'/' + prev
            #1. 형태소 count for Baysian Prior Smoothing
            if word_dict.get(morph) == None:
                word_dict[morph] = 1
            else:
                word_dict[morph] += 1
            #2. 형태소/POS count
            if observ_dict.get(token)== None:
                observ_dict[token] = 1
            else:
                observ_dict[token] += 1
            #3. POS count 
            if POS_dict.get(pos) == None:
                POS_dict[pos] = 1
            else:
                POS_dict[pos] += 1
                
            #4. POS/POS count
            if trans_dict.get(trans_key) == None:
                trans_dict[trans_key] = 1
            else:
                trans_dict[trans_key] += 1
            prev = pos
 
    #II. Bayesian Prior smoothing을 이용한 확률 계산 및 HMM 구축
    trans_list = list(trans_dict.items())
    observ_list = list(observ_dict.items())
    
    POS_total = 0
    word_total = 0
    
    for value in POS_dict.values():
        POS_total += value
    for value in word_dict.values():
        word_total += value
        
    for item in trans_list:
        key = item[0]
        value = item[1]
        p1 = key.split('/')[1]
        p2 = key.split('/')[0]
        prob = math.log((value+POS_dict[p2]/POS_total)/(POS_dict[p1]+1))
        trans_prob_dict[key] = prob
        
    for item in observ_list:
        key = item[0]
        value = item[1]
        p = key.split('/')[1]
        w = key.split('/')[0]
        prob = math.log((value+word_dict[w]/word_total)/(POS_dict[p]+1))
        observ_prob_dict[key] = prob

    return trans_prob_dict, observ_prob_dict

def test_beautifier(test_file):
    #처리가 용이한 형태로 input 변환
    tfp = open(test_file,'r')
    raw_text = tfp.readlines()
    tfp.close()
    beautified = []
    for aLine in raw_text:

        if aLine=='\n':
            continue
        elif len(aLine.split()) != 1:
            temp = []
            temp.append('<s>')
            for token in aLine.split()[1].split('+'):
                temp.append(token)
            beautified.append(temp)
        else:
            beautified.append(['<raw>',aLine])

    return beautified

def calculate_prob(text, trans_dict, observ_dict):
    #Viterbi 알고리즘 활용을 위한 확률값 계산
    prev = '<s>'
    prob = 0.0
    for token in text[1:]:
        pos = token.split('/')[1]
        observ_prob = observ_dict.get(token)
        trans_prob = trans_dict.get(pos+'/'+prev)
        #해당 조합이 없는 경우를 대비한 smoothing
        if observ_prob == None:
            observ_prob = 0.000001
        if trans_prob == None:
            trans_prob = 0.000001
        prob += observ_prob + trans_prob
        prev= pos
    return prob
    
if __name__ == '__main__':    
    train_file = 'train.txt'
    test_file = 'result.txt'
    
    trans_dict, observ_dict = train(train_file)
    test_list = test_beautifier(test_file)

    best_prob = 0.0
    best_result = []
    ofp = open('output.txt','w')
    for text in test_list:
        if text[0] == '<raw>':
            if len(best_result)!=0:
                temp_text = ""
                for parse in best_result[1:-1]:
                    temp_text+=parse+'+'
                temp_text+=best_result[-1]
                print(temp_text,file=ofp)
                print(file=ofp)
                best_prob = 0.0
                best_result = []
            print(text[1],file=ofp)

        else:
            #III. 어절 생성 확률 계산 및 가장 높은 확률을 갖는 형태소 열 탐색
            prob = calculate_prob(text, trans_dict, observ_dict)
            if best_prob ==0.0:
                best_prob = prob
                best_result = text
            elif prob > best_prob:
                best_prob = prob
                best_result = text
    temp_text = ""
    for parse in best_result[1:-1]:
        temp_text+=parse+'+'
    temp_text+=best_result[-1]
    print(temp_text,file=ofp)   
    ofp.close()
    
