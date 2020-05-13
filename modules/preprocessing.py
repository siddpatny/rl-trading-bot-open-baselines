import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

# shows usage to get raw data to input data for training

def getSym(data):
    output = []
    syms = []
    for sym, dataframe in data.groupby('sym'):
        output.append(dataframe)
        syms.append(sym)
    return output, syms

def getMinuteGroup(data):
    output = []
    for df in data:
        time_new = df['time']
        time_idx = pd.DatetimeIndex(time_new)
        grouped = df.groupby([time_idx.hour, time_idx.minute], as_index=False)
        output.append(grouped.mean()[200:1200])
    return output


if __name__ == "__main__":
    df = pd.read_pickle("/u1/data/levels-20191205")
    grouped, syms = getSym(df)
    group = getMinuteGroup(grouped)
    concat = pd.concat(group, axis=1)
    concat.to_csv("concat.csv")
