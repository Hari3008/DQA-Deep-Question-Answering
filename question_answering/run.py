import pickle
import datasets
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
#nltk.download('punkt')
#nltk.download('stopwords')
stpwrds=stopwords.words('english')
import urllib
import pandas as pd
import math
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
print(torch. __version__) 
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.datasets import TUDataset
from sklearn.metrics.pairwise import cosine_similarity
import spacy
nlp = spacy.load('en_core_web_sm') 
from fastcoref import FCoref
from fastcoref import spacy_component
from imblearn.over_sampling import RandomOverSampler

nlp.add_pipe("fastcoref")
coref_model = FCoref()
device = torch.device('cpu')

def get_coref_edges(xx):
  parse = nlp(xx)
  coref_clusters=parse._.coref_clusters
  sentence_spans=[]
  sentences = list(parse.sents)
  sent_span_count=0
  for index,sentence in enumerate(sentences):
    sentence_spans.append((sent_span_count,sent_span_count+len(sentence)-1))
    sent_span_count+=len(sentence)
  coref_edges={}
  for cluster in coref_clusters:
    sent_nums=[]
    full_sent_nums=[]
    for span in cluster:
      for index,sentence_span in enumerate(sentence_spans):
        if(span[0]>=sentence_span[0] and span[1]<sentence_span[1]):
          if(index not in sent_nums):
            sent_nums.append(index)
          full_sent_nums.append(index)
          break
    for index in range(len(sent_nums)-1):
      coref_edges[(sent_nums[index+1],sent_nums[index])]=coref_edges.get((sent_nums[index+1],sent_nums[index]),0)+1  
  return(coref_edges)

def get_answer(question,context,model):
  coref_edges=get_coref_edges(str(question)+". "+str(context))
  out=[str(sent) for sent in nlp(str(question)+". "+str(context)).sents]
  mod = SentenceTransformer('all-MiniLM-L6-v2')
  embedding = mod.encode(out)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  edges=[]
  edges_f=[]

  for index1 in range(len(out)):
    for index2 in range(index1+1,len(out)):
      s1=re.sub('[\'\"\.\(\),\[\]=\{\}]',"",str(out[index1]))
      s2=re.sub('[\'\"\.\(\),\[\]=\{\}]',"",str(out[index2]))
      s1=word_tokenize(s1)
      s2=word_tokenize(s2)

      similar_words=0
      for w1 in s1:
        if(w1.lower() in stpwrds or len(w1)<=2):
          continue
        for w2 in s2:
          if(w2.lower() in stpwrds or len(w2)<=2):
            continue
          if(w1.lower()==w2.lower()):
            similar_words+=1
      ratio=similar_words/max(len(s1),len(s2))
      e1=embedding[index1]
      e2=embedding[index2]
      edge_feature=[]
      if(cosine_similarity([e1], [e2])[0][0]>0.45):
        edge_feature.append(cosine_similarity([e1], [e2])[0][0])
      else:
        edge_feature.append(0)
      if(ratio>0.3):
        edge_feature.append(ratio)
      else:
        edge_feature.append(0)
      edge_feature.append(abs(index1-index2)/len(out))
      if(index1==0 or index2==0):
        edge_feature.append(1)
      else:
        edge_feature.append(0)
      if(abs(index1-index2)==1):
        edge_feature.append(1)
      else:
        edge_feature.append(0) 
      if((index1,index2) in coref_edges):
        edge_feature.append(coref_edges[(index1,index2)]/sum(coref_edges.values()))
      elif((index2,index1) in coref_edges):
        edge_feature.append(coref_edges[(index2,index1)]/sum(coref_edges.values()))
      else:
        edge_feature.append(0)
      if(index1!=0 and index2!=0):
        for index in range(len(edge_feature)):
          edge_feature[index]/=2
      edges.append([index1,index2])
      edges.append([index2,index1])
      edges_f.append(edge_feature)
      edges_f.append(edge_feature)

  x=torch.tensor(embedding,dtype=torch.float)
  edge_index=torch.tensor(edges,dtype=torch.long).t().contiguous()
  edge_attr=torch.tensor(edges_f,dtype=torch.float)
  data=Data(x=x, edge_index=edge_index,edge_attr=edge_attr)
  data = data.to(device)
  oo=model(data.x, data.edge_index, data.edge_attr)
  preds=[]
  preds.append((oo > 0).float().cpu())
  final_answer=""
  for index,pred in enumerate(oo.tolist()):
    if(pred[0]>0):
      final_answer+=out[index]
  return final_answer


input_file=open('./input.txt','r')
input=input_file.read()

q=re.search('question : (.*)\\n',input).group(1).strip()
c=re.search(r'context : (.*)$',input,re.DOTALL).group(1).strip()

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_recall_fscore_support,precision_score,roc_auc_score
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

train_dataset=None

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(train_dataset.num_features, 256, heads=4, edge_dim=6)
        self.lin1 = torch.nn.Linear(train_dataset.num_features, 4 * 256)
        self.conv2 = GATConv(4 * 256, 256, heads=4,  edge_dim=6)
        self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
        self.conv3 = GATConv(4 * 256, train_dataset.num_classes, heads=6,concat=False,  edge_dim=6)
        self.lin3 = torch.nn.Linear(4 * 256, train_dataset.num_classes)

    def forward(self, x, edge_index, edge_attr):
        #print(edge_attr)
        #edge_attr = torch.cat([edge_attr[:, :1], edge_attr[:, 4:]], dim=1)
        #print(edge_attr)
        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr) + self.lin1(x)) 
        x = F.elu(self.conv2(x, edge_index, edge_attr=edge_attr) + self.lin2(x))
        x = self.conv3(x, edge_index, edge_attr=edge_attr) + self.lin3(x)
        return x

inference_model=pickle.load(open('./trained_models/explain.pkl', 'rb'))

#print(q,c)

print()
print()
print('\n\n\nAnswer :\n\n',get_answer(q,c,inference_model))
print()