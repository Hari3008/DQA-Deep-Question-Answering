import spacy 
nlp = spacy.load('en_core_web_sm') 
import pandas as pd
from tqdm import tqdm
import re
import nltk
import sys
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#nltk.download("wordnet")
#nltk.download('omw-1.4')
from heapq import nlargest
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

df=pd.read_csv("../combined_dataset.csv")

d={'question':'Question','answer':'Answer','context':"Context"}
df.rename(columns={k: v for k, v in d.items() if k in df.columns}, inplace=True,errors = "raise")
jcs=0
avg_entities_question=0
avg_entities_context=0
avg_entities_answer=0
avg_words_question=0
avg_words_answer=0
avg_words_context=0
avg_sentences_question=0
avg_sentences_answer=0
avg_sentences_context=0
top_10_question={}
top_10_answer={}
top_10_context={}
le=0

def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union

for index,row in tqdm(df.iterrows(),total=len(df)):
    context_doc=nlp(str(row['Context']))
    context_tokens_filtered=[]
    exclude=["VERB", "ADJ","AUX","PRON", "SPACE", "SCONJ", "ADV", "ADP",'CONJ','CCONJ','DET','INTJ','PART','PUNCT']
    for tok in context_doc:
        if(re.findall("^[a-zA-Z]+$",str(tok),re.MULTILINE) and str(tok).lower() not in stop_words and len(str(tok))>2):
            top_10_context[str(tok).lower()]=top_10_context.get(str(tok).lower(),0)+1
        if(re.findall("^[a-zA-Z0-1]*$",str(tok),re.MULTILINE) and str(tok).lower() not in stop_words and tok.pos_ not in exclude):
            context_tokens_filtered.append(lemmatizer.lemmatize(str(tok).lower()))

    question_doc=nlp(str(row['Question']))
    question_tokens_filtered=[]
    for tok in question_doc:
        if(re.findall("^[a-zA-Z]+$",str(tok),re.MULTILINE) and str(tok).lower() not in stop_words and len(str(tok))>2):
            top_10_question[str(tok).lower()]=top_10_question.get(str(tok).lower(),0)+1
        if(re.findall("^[a-zA-Z0-1]*$",str(tok),re.MULTILINE) and str(tok).lower() not in stop_words and tok.pos_ not in exclude):
            question_tokens_filtered.append(lemmatizer.lemmatize(str(tok).lower()))
    answer_doc=nlp(str(row['Answer']))
    answer_tokens_filtered=[]
    for tok in answer_doc:
        if(re.findall("^[a-zA-Z]+$",str(tok),re.MULTILINE) and str(tok).lower() not in stop_words and len(str(tok))>2):
            top_10_answer[str(tok).lower()]=top_10_answer.get(str(tok).lower(),0)+1
        if(re.findall("^[a-zA-Z0-1]*$",str(tok),re.MULTILINE) and str(tok).lower() not in stop_words and tok.pos_ not in exclude):
            answer_tokens_filtered.append(lemmatizer.lemmatize(str(tok).lower()))
    #print(set(context_tokens_filtered))
    #print(set(question_tokens_filtered))
    #print(set(answer_tokens_filtered))
    jcs+=jaccard_similarity(list(set(question_tokens_filtered)),list(set(context_tokens_filtered)))
    avg_entities_question+=len(question_tokens_filtered)
    avg_entities_context+=len(context_tokens_filtered)
    avg_entities_answer+=len(answer_tokens_filtered)
    avg_words_question+=len(question_doc)
    avg_words_answer+=len(answer_doc)
    avg_words_context+=len(context_doc)
    avg_sentences_question+=len(list(question_doc.sents))
    avg_sentences_answer+=len(list(answer_doc.sents))
    avg_sentences_context+=len(list(context_doc.sents))
    le+=1
    if(index%100==0):
        print(end="\r")
        sys.stdout.write("\033[K")
        print("  ",jcs/(index+1))
    
    #break

print("average jaccard-similarity between question and context:",jcs/le)
print("average entities/question :",avg_entities_question/le)
print("average entities/context :",avg_entities_context/le)
print("average entities/answer :",avg_entities_answer/le)
print("average words/question :",avg_words_question/le)
print("average words/context :",avg_words_context/le)
print("average words/answer :",avg_words_answer/le)
print("average sentences/question :",avg_sentences_question/le)
print("average sentences/context :",avg_sentences_context/le)
print("average sentences/answer :",avg_sentences_answer/le)
print("top 10 words in questions :",nlargest(10, top_10_question, key = top_10_question.get))
print("top 10 words in answers :",nlargest(10, top_10_answer, key = top_10_answer.get))
print("top 10 words in contexts :",nlargest(10, top_10_context, key = top_10_context.get))

