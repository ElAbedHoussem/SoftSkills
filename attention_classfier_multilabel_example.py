import anuvada
import numpy as np
import torch
import pandas as pd
from anuvada.models.classification_attention_rnn import AttentionClassifier
from anuvada.datasets.data_loader import CreateDataset
from anuvada.datasets.data_loader import LoadData
data = CreateDataset()
df = pd.read_csv('./data/ambiguous_test_123_plain.csv',encoding='utf-8')
df.head()
# passing only the first 512 samples, don't have a GPU here!
y = list(df.soft_skill.values)#[0:311]
# print(df.soft_skill.values)
# print('**************************')
# print(len(list(df.soft_skill.values)))
x = list(df.context.values)#[0:311]
# In case of multilabel classification, the labels are expected to be separated by '__'.
x, y, token2id, label2id, lengths_mask = data.create_dataset(x,y, folder_path='./datarnn/', max_doc_tokens=500,multilabel=True,word2vec=True)
# ### Loading created dataset
l = LoadData()
x, y, token2id, label2id, lengths_mask , class_counts= l.load_data_from_path('./datarnn/')
id2token = {v: k for k, v in token2id.iteritems()}
# ### Change into torch vectors
x = torch.from_numpy(x)
y = torch.from_numpy(np.array(y))
# ### Create attention classifier
# You have to pass multilabel loss function as a parameter.
from torch.nn import MultiLabelSoftMarginLoss
acf = AttentionClassifier(vocab_size=len(token2id)+1,embed_size=25,gru_hidden=25,n_classes=len(label2id))
loss = acf.fit(x,y, lengths_mask ,epochs=2,validation_split=0.5,loss=MultiLabelSoftMarginLoss(),multilabel=True)
# ### Predicting
# Prediction needs more than one minibatch at this moment.
acf.predict(x[0:128], lengths_mask[0:128])
# Getting attention
acf.get_attention(x[0:64], lengths_mask[0:64])
# Visualize attention for first sample in the batch 
acf.visualize_attention(x[1:65],lengths_mask[1:65], id2token, './datarnn/visualisation/visual.html')
from IPython.core.display import display, HTML
with open('./datarnn/visualisation/visual.html','r') as f:
    text = f.read()
    display(HTML(text))