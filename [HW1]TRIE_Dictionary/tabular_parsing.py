def grammar_finder(text_list): # 입력 예제를 통해 좌우 접속 정보 분석
    result_set=set()
    for text in text_list:
        parsed = text.split('\n')[0].split('+')
        for i in range(len(parsed)):
            if i==0:
                temp = ('없음',parsed[i].split('/')[1])
            else:
                temp = (parsed[i-1].split('/')[1],parsed[i].split('/')[1])
            result_set.add(temp)
    return result_set

#1. 좌우 접속 정보 분석

raw_fp = open('manual_tagging.txt','r')
raw_text = raw_fp.readlines()
raw_fp.close()

grammar_set = grammar_finder(raw_text)
gr_fp = open('grammar.txt','w')
print('1. 좌우 접속 정보')
print()
for info in grammar_set:
    print('{} - {}'.format(info[0],info[1]))
    print('{} - {}'.format(info[0],info[1]),file=gr_fp)
gr_fp.close()

def trie_constructor(text_list): # TRIE 사전 생성
    result_set=dict()
    temp_list = []
    for text in text_list:
        parsed = text.split('\n')[0].split('+')
        for token in parsed:
            word = token.split('/')[0]
            pos = token.split('/')[1]
            
            temp_list.append((word,pos))

    temp_list.sort(key= lambda word: word[0])
    root_list = []
    for word in temp_list:
        idx = 0
        cur = root_list
        for i in range(len(word[0])):
            flag=False
            for j in range(len(cur)):
                if word[0][i] == cur[j][0]:
                    if i == len(word[0])-1 and word[1] not in cur[j][1]:
                        cur[j][1].append(word[1])
                    flag=True
                    cur = cur[j][2]
                    break
            if flag == False:
                if i == len(word[0])-1:
                    new_set = [word[0][i],[word[1]],[]]
                else:
                    new_set = [word[0][i],['None'],[]]
                    
                cur.append(new_set)
                cur=new_set[2]
            else:
                flag = False

    return root_list

def trie_check(target,trie_set):
    cur=trie_set
    pos=['None']
    for i in range(len(target)):
        pos_flag = False
        for candidate in cur:
            if target[i]==candidate[0]:
                if i == len(target)-1:
                    pos = candidate[1]  
                cur=candidate[2]
                break
    if pos == ['None']:
        return None

    return [target,pos]

def grammar_check(target):
    for elem in grammar_set:
        if elem == target:
            return True
    return False

def trie_parser(token, prior, trie_set):
    #1. 통째로 하나의 형태소인 경우
    pos_list = trie_check(token, trie_set)
    if pos_list!=None:
        for pos in pos_list[1]:
            if pos!= 'None' and grammar_check((prior,pos)):
                return [[token,pos]]
    #2, 2개로 쪼개지는 경우
    if len(token) > 1:
        for i in range(1,len(token)):
            token1 = token[0:i]
            token2 = token[i:]
            pos1_list = trie_check(token1, trie_set)
            pos2_list = trie_check(token2, trie_set)
            
            if pos1_list != None and pos2_list!=None:
                for pos1 in pos1_list[1]:
                    for pos2 in pos2_list[1]:
                        if pos1!='None' and pos2!='None':
                            if grammar_check((prior,pos1)) and grammar_check((pos1,pos2)):

                                return [[token1,pos1],[token2,pos2]]
    #3. 3개로 쪼개지는 경우
    if len(token) > 2:    
        for i in range(1, len(token)-1):
            for j in range(2,len(token)):
                token1 = token[0:i]
                token2 = token[i:j]
                token3 = token[j:]
                pos1_list = trie_check(token1, trie_set)
                pos2_list = trie_check(token2, trie_set)
                pos3_list = trie_check(token3, trie_set)
                if pos1_list != None and pos2_list!=None and pos3_list!=None:
                    for pos1 in pos1_list[1]:
                        for pos2 in pos2_list[1]:
                            for pos3 in pos3_list[1]:
                                print(pos1,pos2,pos3)
                                if pos1!='None' and pos2!='None' and pos3!='None':
                                    if grammar_check((prior,pos1)) and grammar_check((pos1,pos2)) and grammar_check((pos2,pos3)):
                                        return [[token1,pos1],[token2,pos2],[token3,pos3]]
                  

#2. TRIE 사전 생성
trie_set = trie_constructor(raw_text)
print()
print('2. 생성된 TRIE 사전')
print()
print(trie_set)

#3. 예문 형태소 분석
test_fp = open('test.txt','r')
test_text = test_fp.readlines()
test_fp.close()
print()
print('3. 형태소 분석 결과')
print()

for text in test_text:
    print('[원본 문장]')
    print(text)
    print()  
    result = []
    prior='없음'
    temp_text = text.replace('.','').replace('\n','').split(' ')
    print('[형태소 분석 결과]')
    for token in temp_text:
        parsed = trie_parser(token,prior,trie_set)
        print(parsed)
        for elem in parsed:
            prior = elem[1]
            result.append(elem)
    print()
    print('[최종 결과]')          
    for i in range(len(result)):
        print('{}/{}'.format(result[i][0],result[i][1]), end='')
        if i != len(result)-1:
            print('+',end='')
        else:
            print('.\n')
            print()
        
              
    
        

                
