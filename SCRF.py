import torch
import torch.nn as nn


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
                self.Wr=nn.Linear(self.scrf_dim,self.scrf_dim)
                
                self.Gl=nn.Linear(self.scrf_dim,3*self.scrf_dim)
                self.Gr=nn.Linear(self.scrf_dim,3*self.scrf_dim)
                
                self.to_tags=nn.Linear(self.scrf_dim,len(self.tag_to_int))
                self.Dense=nn.Linear(self.word_dim,self.scrf_dim)
                
                self.transitions=nn.Parameter(torch.zeros(self.n_tags,self.n_tags))
                
        def forward(self,feats,tags):
            
                feats=self.Dense(feats)
                scores=self.compute_scores(feats)
                
                gold_socre=self.compute_gold_score(scores,tags)
                normalization_factor=self.forward_pass(scores)
                
                return normalization_factor-gold_socre
                
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
                logalpha=torch.FloatTensor(self.batch_size,scores.shape[1]+1,self.n_tags).fill_(-10000.)
                logalpha[:,0,self.start_id]=0
                
                sen_len=scores.shape[1]
                istarts=[0]*self.allowed_span+[x for x in range(sen_len-self.allowed_span+1)]
                
                for i in range(1,sen_len):
                        tmp=scores[:,istarts[i]:i,i-1]+logalpha[:,istarts[i]:i].unsqueeze(3).expand(self.batch_size,i-istarts[i],self.n_tags,self.n_tags)
                        tmp=tmp.transpose(1,3).contiguous().view(self.batch_size,self.n_tags,self.n_tags*(i-istarts[i]))
                        
                        max_tmp,_=torch.max(tmp,dim=2)
                        tmp=tmp-max_tmp.view(self.batch_size,self.n_tags,1)
                        
                        logalpha[:,i]=max_tmp+torch.log(torch.sum(torch.exp(tmp),dim=2))
                        
                logalpha=logalpha.squeeze()
                alpha=logalpha[-1,:self.stop_id].sum()
                
                return alpha
            
        def compute_gold_score(self,scores,tags):
                
                batch_size=scores.shape[0]
                sen_len=scores.shape[1]
                tagset_size=scores.shape[3]
                
                goldfactors=tags[:,:,0]*sen_len*tagset_size*tagset_size+tags[:,:,1]*tagset_size*tagset_size+tags[:,:,2]*tagset_size+tags[:,:,3]
                factorexprs=scores.view(batch_size,-1)
                
                val=torch.gather(factorexprs,1,goldfactors).sum()
                
                return val