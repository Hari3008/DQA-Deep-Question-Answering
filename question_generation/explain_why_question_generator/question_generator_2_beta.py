import re
import pandas as pd
import nltk
import sys
nltk.download('punkt')
import spacy
from tqdm import tqdm
nlp = spacy.load('en_core_web_sm') 
#from fastcoref import FCoref
#from fastcoref import spacy_component
#nlp.add_pipe("fastcoref")
import os

textbook_pdf_paths = []

for entry in os.listdir('./'):
    if(entry.endswith("prsd.txt")):
        textbook_pdf_paths.append("./"+entry)

df = pd.DataFrame(columns=["Question", "Answer", "Context"],dtype=object)
df_index = 0

#outf = open("questions.txt", "w")

f = open(sys.argv[1], "r")

textbook = f.read()

textbook = re.sub("\n\n", "\n", textbook)
#textbook = re.sub("\n([0-9]+.)*[0-9]\n", "\n", textbook)
#textbook = re.sub("\nFigure.*\n", "\n", textbook)
#textbook = textbook.encode("ascii", errors="ignore").decode()
#textbook = re.sub("\n[ ]", "\n", textbook)
#textbook = re.sub("\n.[0-9]+\n", "\n", textbook)
#textbook = re.sub("\nChapter.*\n", "\n", textbook)
#textbook = re.sub("^\\s*U\\+\\w+\\s*", "", textbook)
#textbook = re.sub("\(.*\)", "", textbook)
# ff=open("corrected_textbook.txt","w")
# ff.write(textbook)


words = ["Hence", "As a result,", "Therefore,", "Thus,"]
question_count = 0
paragraphs=textbook.split("\n")

for word in words:
    for index,paragraph in enumerate(tqdm(paragraphs)):
        for match in re.finditer(word, paragraph):
            start = match.span(0)[0]
            end = match.span(0)[1]

            coreference_text=""
            if(index>1):
                coreference_text+=paragraphs[index-2]
            if(index>0):
                coreference_text+=paragraphs[index-1]
            
            answer=""
            if(start>0):
                answer = paragraph[:start-1]
            coreference_text+=answer
            question = "Explain why"
            context = ""
            extra_para_used=False


            if((len(nltk.sent_tokenize(answer)) <= 2) and (index>0)):
                extra_para_used=True
                answer=paragraphs[index-1]+answer

            x = end
            while x < len(paragraph):
                if paragraph[x] == ".":
                    break
                else:
                    question += paragraph[x]
                x += 1

            coreference_text+=" "+question

            if(extra_para_used and index>1):
                context+=paragraphs[index-2]
            if(index>0):
                context+=paragraphs[index-1]
            context+=paragraph
            if(index<len(paragraphs)-1):
                context+=paragraphs[index+1]

            question_count += 1
            answer += " Hence," + question[12:] + "."
            # context=context[:-12]
            if(len(nltk.sent_tokenize(answer))>6):
                continue

            '''docs = nlp.pipe([coreference_text], component_cfg={"fastcoref": {'resolve_text': True}})
            resolved_text=next(docs)._.resolved_text
            new_question=resolved_text[resolved_text.find("Explain why"):]

            if(new_question!=question):
                docs = nlp.pipe([question], component_cfg={"fastcoref": {'resolve_text': True}})
                resolved_question=next(docs)._.resolved_text
                if(new_question==resolved_question):
                    new_question=question
                    print("Waste : ",resolved_question)
            if(new_question!=question):
                print(question)
                print(resolved_question)
                print(new_question)'''
            
            #new_question=question
            #outf.write("\n\nQuestion : " + question)
            #outf.write("\n\nAnswer : " + answer)
            #outf.write("\n\nContext : " + context)

            #df.loc[df_index] = [question, answer, context]
            df_index += 1
            df=df.append({'Question':question,'Answer':answer,'Context':context},ignore_index = True)

print("Successfully generated ", df_index, " questions")

df.to_csv(sys.argv[2])
