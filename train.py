import torch.optim as optim
import gensim.models as gs

from data_loader import read_lines,read_corpus,DataLoader
from model import NERModel

def train(model,dataLoader,optimizer,epochs):
        
        for epoch in range(epochs):
                loss_final=0
                optimizer.zero_grad()
                
                for _ in range(dataLoader.data_len):
                        x,tags=dataLoader.load_next()

                        loss=model(x,tags)
                        loss_final+=loss
                
                print('epoch=',epoch+1,'training loss=',loss_final.item())
                
                loss_final.backward()
                optimizer.step()
                
                
corpus=read_lines('/eng.txt')
datax,datay,tag_to_int=read_corpus(corpus)

embed_size=50
scrf_size=100
allowed_span_length=6
epochs=20

model=NERModel(embed_size,scrf_size,tag_to_int,tag_to_int['<STOP>'],tag_to_int['<START>'],allowed_span_length)

optimizer=optim.Adagrad(model.parameters(),lr=0.3)

word_dict=gs.Word2Vec(datax,min_count=1,size=embed_size)

data_loader=DataLoader(word_dict,datax,datay)

train(model,data_loader,optimizer,epochs)