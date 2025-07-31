import pandas as pd
import os
files=os.listdir("./")
total_df=pd.DataFrame()

for file in files:
    if(file=="dataset_combiner.py"):
        continue
    total_df=pd.concat([total_df,pd.read_csv(file)])

total_df=total_df.drop_duplicates()
print(total_df)

total_df.to_csv('complete_dataset.csv',index=False)