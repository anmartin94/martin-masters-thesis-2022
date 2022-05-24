import sys
import pandas as pd

filepath = sys.argv[1]
print(filepath)
newfile = "updated_"+filepath
df = pd.read_csv(filepath, sep="\t")
print(df)
df.loc[df["local_pct"]<=.25,["local_pct"]]=1.1
df.loc[df["local_pct"]<=.5,["local_pct"]]=2
df.loc[df["local_pct"]<=.75,["local_pct"]]=3
df.loc[df["local_pct"]<=1,["local_pct"]]=4
df.loc[df["local_pct"]==1.1,["local_pct"]]=1
df.loc[df["global_pct"]<=.25,["global_pct"]]=1.1
df.loc[df["global_pct"]<=.5,["global_pct"]]=2
df.loc[df["global_pct"]<=.75,["global_pct"]]=3
df.loc[df["global_pct"]<=1,["global_pct"]]=4
df.loc[df["global_pct"]==1.1,["global_pct"]]=1
df.to_csv(newfile, sep="\t")

