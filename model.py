import torch.nn as nn
from SCRF import SCRF


class NERModel(nn.Module):
        def __init__(self,embed_dim,scrf_dim,tag_to_int,stop_id,start_id,allowed_span_length):
                super(NERModel,self).__init__()
                
                self.embed_dim=embed_dim
                self.scrf=SCRF(2*embed_dim,scrf_dim,tag_to_int,stop_id,start_id,allowed_span_length)
                
                self.lstm=nn.LSTM(self.embed_dim,self.embed_dim,num_layers=2,bidirectional=True,batch_first=True)
                
        def forward(self,input,tags):

                input,hidden=self.lstm(input,None)
                
                loss=self.scrf(input,tags)
                
                return loss

        def test(self,input):

                output,hidden=self.lstm(input,None)
                return output