import pandas as pd

df=pd.read_csv("../combined_dataset.csv")

error_count=0

for index,row in df.iterrows():
    if(str(row['Question']).startswith("Explain") or str(row['Question']).startswith("Describe") or str(row['Question']).startswith("Justify") or str(row['Question']).startswith("What") or str(row['Question']).startswith("List") or str(row['Question']).startswith("what") or str(row['Question']).startswith("describe") or str(row['Question']).startswith("State") or str(row['Question']).startswith("How") or str(row['Question']).startswith("Who") or str(row['Question']).startswith("Which") or str(row['Question']).startswith("When") or str(row['Question']).startswith("Distinguish") or str(row['Question']).startswith("explain") or str(row['Question']).startswith("Differentiate")):
        pass
    else:
        error_count+=1
        #print(row['Question'])

print(error_count,"bad questions present")
original_length=len(df)
df=df.drop_duplicates()
new_length=len(df)
print(original_length-new_length,"duplicate question present")