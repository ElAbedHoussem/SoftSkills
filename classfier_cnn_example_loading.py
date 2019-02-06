import anuvada
import numpy as np
import torch
import pandas as pd
from anuvada.datasets.data_loader import CreateDataset
from anuvada.datasets.data_loader import LoadData
from anuvada.models.classification_cnn import ClassificationCNN

l = LoadData()
x, y, token2id, label2id, lengths_mask, class_counts = l.load_data_from_path('./data/')
id2token = {v: k for k, v in token2id.iteritems()}
# ### Change into torch vectors
x = torch.from_numpy(x)
y = torch.from_numpy(np.array(y))
len(id2token)
# ### Create attention classifier
acf = ClassificationCNN(vocab_size=len(token2id),embed_dim=300,num_classes=len(label2id),
                        word2vec_path='./data/word2vec_300_5_5')

from anuvada.utils import save_model, load_model
load_model(acf, './data/model_epoch1.pth')
loss = acf.fit(x,y,batch_size=64,epochs=1,validation_split=0.2,run_on='cpu')
# ### Predicting
# Prediction needs more than one minibatch at this moment.
# print(x[-10])
predictions, df = acf.compute_saliency_map(x[10],1,'visual.html',id2token)#-3674
from IPython.core.display import display, HTML
print(predictions)
with open('visual.html','r') as f:
    text = f.read()
    display(HTML(text))
# predictions, df = acf.compute_saliency_map(x[-20],1,'visual.html',id2token)
# print(predictions)

# with open('visual.html','r') as f:
#     text = f.read()
#     display(HTML(text))
# predictions, df = acf.compute_saliency_map(x[-30],1,'visual.html',id2token)
# print(predictions)
# with open('visual.html','r') as f:
#     text = f.read()
#     display(HTML(text))

# predictions, df = acf.compute_saliency_map(x[-40],1,'visual.html',id2token)
# print(predictions)
# with open('visual.html','r') as f:
#     text = f.read()
#     display(HTML(text))

