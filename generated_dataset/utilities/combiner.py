import pandas as pd
import os

total_df=pd.DataFrame()
for file in os.listdir("./"):
    try:
        print(file)
        if(file in ["combiner.py","combined_dataset.csv"]):
            continue
        df=pd.read_csv(file)
        if('Question' not in df.columns):
            df=df[['question','answer','context']]
            df.rename(columns = {'question':'Question'}, inplace = True)
            df.rename(columns = {'answer':'Answer'}, inplace = True)
            df.rename(columns = {'context':'Context'}, inplace = True)
        else:
            df=df[['Question','Answer','Context']]
        total_df=pd.concat([total_df,df])
    except Exception as e:
        print(e)
total_df.to_csv("combined_dataset.csv",index=False)
total_df=total_df.drop_duplicates()
print("Total number of questions : ",len(total_df))