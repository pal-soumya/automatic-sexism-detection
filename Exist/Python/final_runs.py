import pandas as pd
import numpy as np

def final(filepath, saveas):
    df = pd.read_csv(filepath, sep="\t", header = None) 
    ids = df[1].apply(lambda x : str(x).zfill(6))
    df[1] = ids
    pd.DataFrame(df).to_csv(saveas, sep = "\t", index=False,header=False)
    return df

import os


file = os.path.join(os.path.expanduser("~"),'Downloads/TASK_1_EN_EVALL_TSV_REPORT.tsv')
df = pd.read_csv(file, sep="\t", header = None) 