import torch
import torch.nn as nn
import numpy as np


class SCRF(nn.Module):
        def __init__(self,word_dim,scrf_dim,tag_to_int,stop_id,start_id,allowed_span_length):
                super(SCRF,self).__init__()
            
                self.tag_to_int=tag_to_int
                self.stop_id=stop_id
                self.start_id=start_id
                self.n_tags=len(tag_to_int)
                
                self.allowed_span=allowed_span_length
                self.batch_size=1
                
                self.word_dim=word_dim
                self.scrf_dim=scrf_dim
                
                self.Wl=nn.Linear(self.scrf_dim,self.scrf_dim)
                self.init_linear(self.Wl)

                self.Wr=nn.Linear(self.scrf_dim,self.scrf_dim)
                self.init_linear(self.Wr)
                
                self.Gl=nn.Linear(self.scrf_dim,3*self.scrf_dim)
                self.init_linear(self.Gl)

                self.Gr=nn.Linear(self.scrf_dim,3*self.scrf_dim)
                self.init_linear(self.Gr)

                self.to_tags=nn.Linear(self.scrf_dim,len(self.tag_to_int))
                self.init_linear(self.to_tags)
                
                self.Dense=nn.Linear(self.word_dim,self.scrf_dim)
                self.init_linear(self.Dense)

                self.transitions=nn.Parameter(torch.randn(self.n_tags,self.n_tags))
                
        def forward(self,feats,tags):
            
                feats=self.Dense(feats)
                scores=self.compute_scores(feats)

                gold_socre=self.compute_gold_score(scores,tags)
                normalization_factor=self.forward_pass(scores)
                
                return normalization_factor-gold_socre

        def init_linear(self, input_linear):
    
                bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
                nn.init.uniform(input_linear.weight, -bias, bias)
                
                if input_linear.bias is not None:
                        input_linear.bias.data.zero_()
                
        def compute_scores(self,feats):
                scores=torch.zeros(self.batch_size,feats.shape[1],feats.shape[1],self.scrf_dim)
                diag=torch.LongTensor(range(feats.shape[1]))
                
                ht=feats
                scores[:,diag,diag]=ht
                
                if feats.shape[1]==1:
                    
                        return self.to_tags(scores).unsqueeze(3)+self.transitions.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    
                for span_len in range(1,min(self.allowed_span,feats.shape[1])):
                        ht_1_l=ht[:,:-1]
                        ht_1_r=ht[:,1:]

                        h_t_hat=4*torch.sigmoid(self.Wl(ht_1_l)+self.Wr(ht_1_r))-2
                      
                        w=torch.exp(self.Gl(ht_1_l)+self.Gr(ht_1_r)).view(self.batch_size,feats.shape[1]-span_len,3,self.scrf_dim).permute(2,0,1,3)
                        w=w/w.sum(0).unsqueeze(0).expand(3,self.batch_size,feats.shape[1]-span_len,self.scrf_dim)
                       
                        
                        ht=w[0]*h_t_hat+w[1]*ht_1_l+w[2]*ht_1_r
                        
                        scores[:,diag[:-span_len],diag[span_len:]]=ht

                return self.to_tags(scores).unsqueeze(3)+self.transitions.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    
        def forward_pass(self,scores):

                logalpha=torch.FloatTensor(self.batch_size,scores.shape[1]+1,self.n_tags).fill_(-1000)
                logalpha[:,0,self.start_id]=0
                
                sen_len=scores.shape[1]
                istarts=[0]*self.allowed_span+[x for x in range(sen_len-self.allowed_span+1)]
                
                for i in range(1,sen_len+1):
                        tmp=scores[:,istarts[i]:i,i-1]+logalpha[:,istarts[i]:i].unsqueeze(3).expand(self.batch_size,i-istarts[i],self.n_tags,self.n_tags)
                        tmp=tmp.transpose(1,3).contiguous().view(self.batch_size,self.n_tags,self.n_tags*(i-istarts[i]))
                        
                        max_tmp,_=torch.max(tmp,dim=2)
                        tmp=tmp-max_tmp.view(self.batch_size,self.n_tags,1)
                        
                        logalpha[:,i]=max_tmp+torch.log(torch.sum(torch.exp(tmp),dim=2))

                mask=torch.tensor([sen_len])        
                mask = mask.unsqueeze(1).unsqueeze(1).expand(self.batch_size, 1, self.n_tags)
                alpha = torch.gather(logalpha, 1, mask).squeeze(0)

                max_l=torch.max(alpha)
                alpha=alpha-max_l

                return max_l+torch.log(torch.sum(torch.exp(alpha)))
 
        def compute_gold_score(self,scores,tags):
                
                batch_size=scores.shape[0]
                sen_len=scores.shape[1]
                tagset_size=scores.shape[3]
                
                goldfactors=tags[:,:,0]*sen_len*tagset_size*tagset_size+tags[:,:,1]*tagset_size*tagset_size+tags[:,:,2]*tagset_size+tags[:,:,3]
                factorexprs=scores.view(batch_size,-1)
                
                val=torch.gather(factorexprs,1,goldfactors).sum()
              
                return val

        def decode(self, scores):
        
                batch_size = scores.size(0)
                sentlen = scores.size(1)

                scores = scores.data

                logalpha = torch.FloatTensor(batch_size, sentlen+1, self.n_tags).fill_(-10000.)
                logalpha[:, 0, self.start_id] = 0.

                starts = torch.zeros((batch_size, sentlen, self.n_tags))
                ys = torch.zeros((batch_size, sentlen, self.n_tags))

                for j in range(1, sentlen + 1):
                        istart = 0

                        if j > self.allowed_span:
                                istart = max(0, j - self.allowed_span)

                        f = scores[:, istart:j, j - 1].permute(0, 3, 1, 2).contiguous().view(batch_size, self.n_tags, -1) + \
                            logalpha[:, istart:j].contiguous().view(batch_size, 1, -1).expand(batch_size, self.n_tags, (j - istart) * self.n_tags)

                        logalpha[:, j, :], argm = torch.max(f, dim=2)
                        starts[:, j-1, :] = (argm / self.n_tags + istart)
                        ys[:, j-1, :] = (argm % self.n_tags)

                batch_scores = []
                batch_spans = []

                for i in range(batch_size):

                        spans = []
                        batch_scores.append(max(logalpha[i,sentlen-1]))

                        end = sentlen-1
                        y = self.stop_id

                        while end >= 0:
                                start = int(starts[i, end, y])
                                y_1 = int(ys[i, end, y])
    
                                spans.append((start, end, y_1, y))
    
                                y = y_1
                                end = start - 1

                        batch_spans.append(spans)

                return batch_spans, batch_scores
      
        def validate(self,sentence,tags):
                sentence=self.Dense(sentence)
                scores=self.compute_scores(sentence)

                gold_score=self.compute_gold_score(scores,tags)
                normalization_factor=self.forward_pass(scores)

                loss=normalization_factor-gold_score

                spans,score=self.decode(scores)
                spans=spans[0]
                spans.reverse()
                tags=tags.squeeze(0)

                gold=[]
                predict=[]

                for span in  tags:
                        for number in range(int(span[0]),int(span[1])+1):
                                gold.append(int(span[3]))

                for span in  spans:
                        for number in range(span[0],span[1]+1):
                                predict.append(span[3])

                count=0
                for i in range(len(gold)):
                        if predict[i]==gold[i]:
                                count+=1

                return loss,100*count/len(gold)