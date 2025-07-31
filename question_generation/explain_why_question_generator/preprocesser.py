import re
import os
import spacy
import sys
from tqdm import tqdm
model= spacy.load('en_core_web_sm') 


file=open(sys.argv[1],'r')
text=file.read()
text=re.sub("[\n]+","\n",text)
out=""

prev="."
for sentence in tqdm(text.split("\n")):
    parse=model(sentence)
    #print([str(x) for x in list(parse)])
    if(str(prev)!="."):
        out+=" "+sentence
    elif(len(parse)>15):
        out+="\n\n"+sentence
    try:
        prev=list(parse)[-1]
    except:
        prev="."

out_file=open(sys.argv[1][:-4]+"_prsd.txt",'w')
out_file.write(out)
