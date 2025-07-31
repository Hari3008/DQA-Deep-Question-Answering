import textstat
import pandas as pd
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

#Choosing SQUAD dataset
#df=pd.read_csv("/home/ausdauerer/Downloads/squad.csv")

#Choosing our dataset
df=pd.read_csv("/home/ausdauerer/Downloads/squad.csv")


d={'question':'Question','answer':'Answer','context':"Context"}
df.rename(columns={k: v for k, v in d.items() if k in df.columns}, inplace=True,errors = "raise")
question_score=0
context_score=0
q_ars=0
c_ars=0
question_scores=[]
context_scores=[]

q_counter=0
c_counter=0
for index,row in tqdm(df.iterrows(),total=len(df)):
    #print(row['Question'],row['Context'])
    question_score+=textstat.flesch_reading_ease(str(row['Question']))
    question_scores.append(math.floor(textstat.flesch_reading_ease(str(row['Question']))))
    q_ars+=textstat.automated_readability_index(str(row['Question']))
    q_counter+=1
    context_score+=textstat.flesch_reading_ease(str(row['Context']))
    context_scores.append(math.floor(textstat.flesch_reading_ease(str(row['Context']))))
    c_ars+=textstat.automated_readability_index(str(row['Context']))
    c_counter+=1


#plt.hist(question_scores)
#plt.hist(context_scores)
#plt.xlim(-200,200)
#plt.show()
print("Question average flesch reading ease score : ",question_score/q_counter)
print("Question average automatic readability index : ",q_ars/q_counter)
print("Context average flesch reading ease score : ",context_score/c_counter)
print("Context average automatic readability index : ",c_ars/c_counter)
