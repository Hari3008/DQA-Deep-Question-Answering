import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk, string
import math

#nltk.download('punkt')

stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]


average_cos_sim=0
count=0

df=pd.read_csv("./complete_dataset.csv")
question_type="Justify"
if(question_type!='all'):
  df=df.loc[df.question.str.startswith(question_type, na=False)]
  df.reset_index(inplace = True)
  df=df[["question", "answer","context"]]
df=df[:math.floor(len(df)*75/100)]

for index,row in df.iterrows():
    try:
        average_cos_sim+=cosine_sim(row['question'], row['context'])
        count+=1
    except:
        pass

average_cos_sim/=count

print(average_cos_sim)