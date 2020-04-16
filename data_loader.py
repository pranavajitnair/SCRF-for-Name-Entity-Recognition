import os
import torch

def read_lines(filename):
        with open(os.getcwd()+filename, 'r') as f:
                corpus = f.readlines()
        
        return corpus

def read_corpus(lines):
        
        features=[]
        labels=[]
        tmp_fl=[]
        tmp_ll=[]
        
        for line in lines:
            
            if not (line.isspace() or (len(line)>10 and line[0:10]=='-DOCSTART-')):
                
                line=line.rstrip('\n').split()
                tmp_fl.append(line[0])
                tmp_ll.append(line[-1])
                
            elif len(tmp_fl)>0:
                
                features.append(tmp_fl)
                labels.append(iob_iobes(tmp_ll))
                tmp_fl=list()
                tmp_ll=list()
                
        if len(tmp_fl)>0:
            
            features.append(tmp_fl)
            labels.append(iob_iobes(tmp_ll))
            
        labels=CRFtag_to_SCRFtag(labels)
    
        tag_to_int={}
        tag_to_int['PER']=0
        tag_to_int['LOC']=1
        tag_to_int['ORG']=2
        tag_to_int['MISC']=3
        tag_to_int['<START>']=4
        tag_to_int['<STOP>']=5
        tag_to_int['O']=6
        
        labels=encode_SCRF(labels,tag_to_int)
        
        return features,labels,tag_to_int

def iob_iobes(tags):
        
        new_tags=[]
        iob2(tags)
        
        for i,tag in enumerate(tags):
            
            if tag=='O':
                new_tags.append(tag)
                
            elif tag.split('-')[0]=='B':
                
                if i+1!=len(tags) and \
                   tags[i+1].split('-')[0]=='I':
                    new_tags.append(tag) 
                else:
                    new_tags.append(tag.replace('B-','S-'))
                    
            elif tag.split('-')[0]=='I':
                
                if i+1<len(tags) and \
                        tags[i+1].split('-')[0]=='I':
                    new_tags.append(tag)  
                else:
                    new_tags.append(tag.replace('I-','E-'))
                    
            else:
                raise Exception('Invalid IOB format!')
                
        return new_tags
    
def iob2(tags):
    
        for i,tag in enumerate(tags):
            
            if tag=='O':
                continue
            split=tag.split('-')
            if len(split)!=2 or split[0] not in ['I','B']:
                return False
            if split[0]=='B':
                continue 
            elif i==0 or tags[i-1]=='O':
                tags[i]='B'+tag[1:]  
            elif tags[i-1][1:]==tag[1:]:
                continue
            else:
                tags[i]='B'+tag[1:]
                
        return True

def CRFtag_to_SCRFtag(inputs):
        
        alltags = []
        
        for input in inputs:
            
                tags = []
                beg = 0
                oldtag = '<START>'
                
                for i, tag in enumerate(input):
                    
                        if tag == u'O':
                                tags.append((i, i, oldtag, tag))
                                oldtag = tag
                        if tag[0] == u'S':
                                tags.append((i, i, oldtag, tag[2:]))
                                oldtag = tag[2:]
                        if tag[0] == u'B':
                                beg = i
                        if tag[0] == u'E':
                                tags.append((beg, i, oldtag, tag[2:]))
                                oldtag = tag[2:]
                                
                alltags.append(tags)
                
        return alltags
    
def encode_SCRF(input_lines, word_dict):
        lines = list(map(lambda t: list(map(lambda m: [m[0], m[1], word_dict[m[2]], word_dict[m[3]]], t)), input_lines))
        
        return lines


class DataLoader(object):
        def __init__(self,word_dict,x_train,y_train,x_test,y_test,x_validate,y_validate):
                self.x_train=x_train
                self.y_train=y_train
                self.data_len=len(x_train)
                
                self.x_test=x_test
                self.y_test=y_test
                self.test_len=len(x_test)

                self.x_valiate=x_validate
                self.y_validate=y_validate
                self.validate_len=len(x_validate)

                self.word_dict=word_dict
                self.counter_train=0
                self.counter_test=0
                self.counter_validate=0
        
        def load_next(self):
                sentence=self.x_train[self.counter_train]
                l=[]
                
                for word in sentence:
                        l.append(self.word_dict[word])
                        
                l=torch.tensor(l).unsqueeze(0)
                y=self.y_train[self.counter_train]
                y=torch.tensor(y).unsqueeze(0)
                
                self.counter_train=(self.counter_train+1)%self.data_len
                
                return l,y

        def load_next_test(self,test):
                if test:
                        sentence=self.x_test[self.counter_test]
                        y=self.y_test[self.counter_test]
                        self.counter_test=(self.counter_test+1)%self.test_len
                else:
                        sentence=self.x_valiate[self.counter_validate]
                        y=self.y_validate[self.counter_validate]
                        self.counter_validate=(self.counter_validate+1)%self.validate_len

                l=[]
                
                for word in sentence:
                        l.append(self.word_dict[word])
                        
                l=torch.tensor(l).unsqueeze(0)
                y=torch.tensor(y).unsqueeze(0)
              
                return l,y