import anuvada
import numpy as np
import torch
import pandas as pd
from anuvada.models.classification_attention_rnn import AttentionClassifier
from anuvada.datasets.data_loader import CreateDataset
from anuvada.datasets.data_loader import LoadData


data = CreateDataset()
df = pd.read_csv('./datar/ambiguous_test_123_plain.csv')
# passing only the first 512 samples, don't have a GPU here!
y = list(df.Genre.values)[0:512]
x = list(df.summary.values)[0:512]
x, y = data.create_dataset(x,y, folder_path='./datarnn/', max_doc_tokens=500)
# ### Loading created dataset
l = LoadData()
x, y, token2id, label2id, lengths_mask = l.load_data_from_path('./datarnn/')
id2token = {v: k for k, v in token2id.iteritems()}
# ### Change into torch vectors
x = torch.from_numpy(x)
y = torch.from_numpy(y)
# ### Create attention classifier
acf = AttentionClassifier(vocab_size=len(token2id)+2,embed_size=25,gru_hidden=25,n_classes=len(label2id))
loss = acf.fit(x,y, lengths_mask ,epochs=1,validation_split=0.5)
# ### Predicting
# Prediction needs more than one minibatch at this moment.
acf.predict(x[0:128], lengths_mask[0:128])
# Getting attention
acf.get_attention(x[0:64], lengths_mask[0:64])
# Visualize attention for first sample in the batch 
acf.visualize_attention(x[1:65],lengths_mask[1:65], id2token, './datarnn/visual.html')
from IPython.core.display import display, HTML
# Everything is noise now, so the attention weights don't make sense.
with open('./datarnn/visual.html','r') as f:
    text = f.read()
    display(HTML(text))
get_ipython().system(' head visual.html')
