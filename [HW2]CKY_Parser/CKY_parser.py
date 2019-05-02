
def grammar_loader():
    raw_grammar = open('grammar.txt','r')
    rule_dict = {}
    for line in raw_grammar.readlines():
        if line != ('\n'):
            value = line.split('->')[0].strip()
            key = line.split('->')[1].strip()
            if rule_dict.get(key) != None:
                rule_dict[key].append(value)
            else:
                rule_dict[key]=[value]
    return rule_dict
        

def parser(target,grammar, ugfp):
    counter = 1
    counter_dict= {}
    
    length = len(target)
    parse_table = [[[] for x in range(length - y)] for y in range(length)]

    for i, word in enumerate(target):
        if grammar.get(word)!=None:
            for pos in grammar.get(word):
                parse_table[0][i].append([grammar.get(pos)[0],[word,None]])
                print("{} ({}, {})".format(counter, pos, word), file=ugfp)
                counter_dict["{}, ({}, {})".format(pos, 0,i)]=counter                
                counter += 1
                print("{} ({}, {})".format(counter, grammar.get(pos)[0], counter_dict["{}, ({}, {})".format(pos, 0,i)]), file=ugfp)
                counter_dict["{}, ({}, {})".format(grammar.get(pos)[0], 0,i)]=counter
                counter += 1
    for index in range(2,length+1):
        for start in range(0,length - index +1):
            for left_size in range(1,index):
                right_size = index - left_size
                left_cell = parse_table[left_size - 1][start]
                right_cell = parse_table[right_size -1][start +left_size]

                candidates=[]
                for left in left_cell:
                    for right in right_cell:
                        candidates.append([left[0]+' '+right[0],(left_size - 1,start),(right_size -1,start +left_size)])
                for candidate in candidates:
                    if grammar.get(candidate[0])!=None:
                        for pos in grammar.get(candidate[0]):
                            #print(candidate)
                            parse_table[index-1][start].append([pos,[candidate[1],candidate[2],candidate[0]]])
                            left_pos = candidate[0].split()[0]
                            right_pos = candidate[0].split()[1]
                            print("{} ({}, ({}, {}))".format(counter,pos,counter_dict[left_pos + ', '+ str(candidate[1])],counter_dict[right_pos + ', '+ str(candidate[2])]), file=ugfp)
                            counter_dict["{}, ({}, {})".format(pos, index-1,start)]=counter
                            counter += 1
                            
    print(file=ugfp)                        
    return parse_table

def print_tree(parsed,ofp):
    roots = [x for x in parsed[-1][0] if x[0] == 'S']

    parse_trees = [generate_tree(parsed, node) for node in roots]
    tree_set = set()
    for tree in parse_trees:
        tree_set.add(tree)
    for tree in tree_set:
        print(tree,file=ofp)
    print(file=ofp)

def generate_tree(parsed, node):
    if node[1][1] == None:
        return "({} {})".format(node[0],node[1][0])
    #추가
    left_nodes = parsed[node[1][0][0]][node[1][0][1]]
    left_pos = node[1][2].split()[0]
    right_nodes = parsed[node[1][1][0]][node[1][1][1]]
    right_pos = node[1][2].split()[1]

    for left in left_nodes:
        if left_pos == left[0]:
            left_node=left
    for right in right_nodes:
        if right_pos == right[0]:
            right_node=right
    return "({} ({} {}))".format(node[0],generate_tree(parsed,left_node),generate_tree(parsed,right_node))
                                                        
if __name__ == '__main__':
    
    grammar = grammar_loader()
    ifp = open('input.txt','r')
    ofp = open('output.txt','w')
    ugfp = open('used_grammar.txt','w')

    for line in ifp.readlines():
        target = line.strip().split()
        parsed = parser(target,grammar,ugfp)
        print_tree(parsed, ofp)
    ifp.close()
    ugfp.close()
    ofp.close()

    
