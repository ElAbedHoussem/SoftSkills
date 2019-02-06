

import anuvada
import numpy as np
import torch
import pandas as pd
from anuvada.datasets.data_loader import CreateDataset
from anuvada.datasets.data_loader import LoadData
from anuvada.models.classification_cnn import ClassificationCNN
data = CreateDataset()
df = pd.read_csv('./data/ambiguous_train_123_begin.tsv',encoding='utf-8',sep='\t')
df.head()
# passing only the first 512 samples, don't have a GPU here!
y = list(df.soft_skill.values)
x = list(df.context.values)
x, y, token2id, label2id, lengths_mask = data.create_dataset(x,y, folder_path='./data/',
                                                             max_doc_tokens=500,word2vec=True,multilabel=False)
# ### Loading created dataset
l = LoadData()
x, y, token2id, label2id, lengths_mask, class_counts = l.load_data_from_path('./data/')
id2token = {v: k for k, v in token2id.iteritems()}
# ### Change into torch vectors
x = torch.from_numpy(x)
y = torch.from_numpy(np.array(y))
len(id2token)
# ### Create attention classifier
# You have to pass multilabel loss function as a parameter.
acf = ClassificationCNN(vocab_size=len(token2id),embed_dim=300,num_classes=len(label2id),
                        word2vec_path='./data/word2vec_300_5_5')
loss = acf.fit(x,y,batch_size=64,epochs=1,validation_split=0.2,run_on='gpu')
from anuvada.utils import save_model
save_model(acf, './data/model_epoch1.pth')
loss = acf.fit(x,y,batch_size=64,epochs=1,validation_split=0.2,run_on='gpu')
save_model(acf, './data/model_epoch2.pth')
loss = acf.fit(x,y,batch_size=64,epochs=1,validation_split=0.2,run_on='gpu')
save_model(acf, './data/model_epoch3.pth')
loss = acf.fit(x,y,batch_size=64,epochs=1,validation_split=0.2,run_on='gpu')
save_model(acf, './data/model_epoch4.pth')
loss = acf.fit(x,y,batch_size=64,epochs=1,validation_split=0.2,run_on='gpu')
save_model(acf, './data/model_epoch5.pth')
loss = acf.fit(x,y,batch_size=64,epochs=1,validation_split=0.2,run_on='gpu')
save_model(acf, './data/model_epoch6.pth')
loss = acf.fit(x,y,batch_size=64,epochs=1,validation_split=0.2,run_on='gpu')
save_model(acf, './data/model_epoch7.pth')
loss = acf.fit(x,y,batch_size=64,epochs=1,validation_split=0.2,run_on='gpu')
save_model(acf, './data/model_epoch8.pth')
loss = acf.fit(x,y,batch_size=64,epochs=1,validation_split=0.2,run_on='gpu')
save_model(acf, './data/model_epoch9.pth')
loss = acf.fit(x,y,batch_size=64,epochs=1,validation_split=0.2,run_on='gpu')
save_model(acf, './data/model_epoch10.pth')
loss = acf.fit(x,y,batch_size=64,epochs=1,validation_split=0.2,run_on='gpu')
save_model(acf, './data/model_epoch11.pth')

# ### Predicting

# Prediction needs more than one minibatch at this moment.
acf.predict(x[0:128])

