import spacy
nlp = spacy.load('en_core_web_sm') 
from fastcoref import FCoref
from fastcoref import spacy_component
nlp.add_pipe("fastcoref")
import pandas as pd
from tqdm import tqdm
import sys

df=pd.read_csv(sys.argv[1])
output_df=None
try:
    output_df=pd.read_csv("combined_dataset_coreference_resolved.csv")
    output_df=output_df[["Question","Answer","Context"]]
    df=df[len(output_df):]
except:
    output_df=pd.DataFrame(columns=["Question", "Answer", "Context"],dtype=object)
    output_df.to_csv("combined_dataset_coreference_resolved.csv")

counter=0
with tqdm(total=len(df)) as pbar:
    for index,row in df.iterrows():
        context=row['Context']
        question_text=row['Question'][13:]
        doc=nlp(context)
        sentences=doc.sents
        coreference_text=""
        for sent in sentences:
            if(question_text in str(sent)):
                break
            else:
                coreference_text+=str(str(sent)+" ")
        coreference_text+=row['Question']
        docs=nlp.pipe([coreference_text], component_cfg={"fastcoref": {'resolve_text': True}})
        resolved_text=next(docs)._.resolved_text
        new_question=resolved_text[resolved_text.find("Explain why"):]
        print("Old question : "+row['Question'])
        print("New question : "+new_question)
        #df.at[index,'Question']=new_question
        output_df=output_df.append({'Question':new_question,'Answer':row['Answer'],'Context':row['Context']},ignore_index = True)
        counter+=1
        if(counter%10==0):
            output_df.to_csv("combined_dataset_coreference_resolved.csv")
        pbar.update(1)

output_df.to_csv("combined_dataset_coreference_resolved.csv")