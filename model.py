import torch.nn as nn
from SCRF import SCRF


class NERModel(nn.Module):
        def __init__(self,embed_dim,scrf_dim,tag_to_int,stop_id,start_id,allowed_span_length):
                super(NERModel,self).__init__()
                
                self.embed_dim=embed_dim
                self.scrf=SCRF(embed_dim*2,scrf_dim,tag_to_int,stop_id,start_id,allowed_span_length)
                
                self.lstm=nn.LSTM(self.embed_dim,self.embed_dim,num_layers=2,bidirectional=True,batch_first=True)
                
        def forward(self,sentence,tags):
                input,hidden=self.lstm(sentence,None)
                
                loss=self.scrf(input,tags)
                
                return loss