import torch.optim as optim
import gensim.models as gs
from data_loader import read_lines,read_corpus,DataLoader
from model import NERModel

def train(model,dataLoader,optimizer,epochs,validate_epoch):

        for epoch in range(epochs):

                model.train()
                loss_final=0
                optimizer.zero_grad()
                
                for _ in range(dataLoader.data_len//100):
                        x,tags=dataLoader.load_next()

                        loss=model(x,tags)
                        loss_final+=loss
                
                t=(dataLoader.data_len//100)
                final_loss=loss_final.item()/t

                loss_final.backward()
                # nn.utils.clip_grad_norm_(model.parameters(),3.0)
                optimizer.step()

                model.eval()
                loss_validation=0
                accuracy=0

                for _ in range(validate_epoch):
                        x,y=dataLoader.load_next_test(False)
                        x=model.test(x)

                        loss,acc=model.scrf.validate(x,y)
                        loss_validation+=loss.item()
                        accuracy+=acc

                print('epoch=',epoch+1,'training loss=',final_loss,'validation loss=',loss_validation/validate_epoch,'validation accuracy=',accuracy/validate_epoch)

def test(model,dataLoader,test_epochs):
        
        test_loss=0
        test_accuracy=0

        for _ in  range(test_epochs):
                x,y=dataLoader.load_next_test(True)
                x=model.test(x)

                loss,acc=model.scrf.validate(x,y)
                test_loss+=loss.item()
                test_accuracy+=acc

        print('test loss=',test_loss/test_epochs,'test accuracy=',test_accuracy/test_epochs)

corpus=read_lines('/eng.txt')
datax,datay,tag_to_int=read_corpus(corpus)

corpus_test=read_lines('/eng_test.txt')
testx,testy,_=read_corpus(corpus_test)

corpus_validate=read_lines('/eng_validate.txt')
validatex,validatey,_=read_corpus(corpus_validate)

embed_size=50
scrf_size=100
allowed_span_length=6
epochs=100
validate_epochs=len(validatex)
test_epochs=len(testx)

model=NERModel(embed_size,scrf_size,tag_to_int,tag_to_int['<STOP>'],tag_to_int['<START>'],allowed_span_length)

optimizer=optim.Adagrad(model.parameters(),lr=0.009)

word_dict=gs.Word2Vec(datax+validatex+testx,min_count=1,size=embed_size)

data_loader=DataLoader(word_dict,datax,datay,testx,testy,validatex,validatey)

train(model,data_loader,optimizer,epochs,validate_epochs)

test(model,data_loader,test_epochs)