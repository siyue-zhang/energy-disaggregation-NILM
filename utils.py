
import os
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import torch

rcParams['figure.figsize'] = (13, 6)

# System

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device

def save_file(data, fname, fpath="./"):
    with open(fpath + fname+ f".pkl", 'wb') as f:
        pickle.dump(data, f)
    f.close()
    return

def load_file(fname, fpath="./"):
    with open(fpath + fname+ f".pkl", 'rb') as f:
        data = pickle.load(f)
    f.close()
    return data

# Timestamp

def str_to_ts(str_list, format="%Y-%m-%d %H:%M:%S"):
    ts = [datetime.strptime(str, format) for str in str_list]
    return ts

def ts_to_str(ts_list, format="%Y-%m-%d %H:%M:%S"):
    str = [ts.strftime(format) for ts in ts_list]
    return str

# String operation

def str_to_int(str):
    out = []
    for ch in str:
        if ch.isnumeric():
            out.append(ch)
    return int("".join(out))

# Dataframe operation

def combine_files(folder_path, col_time):
    files = os.listdir(folder_path)
    if len(files)<1:
        return None
    else:
        for file in files:
            path = folder_path + "/" + file

            if file==files[0]:
                df = pd.read_csv(path)
            else:
                df2 = pd.read_csv(path)
                df = pd.concat([df,df2], sort=False, ignore_index=True)
        is_col_na = df[col_time].isna()
        not_na = is_col_na==False
        no_na_index = df.index[not_na]
        df = df.loc[no_na_index]
        return df

def count_na_dict(df):
    counts = df.isna().sum()
    return counts.to_dict()

def count_outlier_dict(df, min=None, max=None):
    # min and max are not considered as outlier values
    # na is not considered as outlier value
    if (min and max):
        assert(max>min)
    if min and not max:
        flg = df<min
        counts = flg.sum().to_dict()
    if max and not min:
        flg = df>max
        counts = flg.sum().to_dict()
    if max and min:
        flg1 = df<min
        flg2 = df>max
        counts = flg1.sum()+flg2.sum()
        counts = counts.to_dict()
    return counts 

def fill_na_by_interpolate(df):
    df = df.interpolate()
    return df

def resample_df(df, freq):
    df = df.resample(f"{freq}T").mean()
    return df

def show_visualization(data, show_n_features):

    feature_keys = data.columns
    show_n_features = min(show_n_features, len(feature_keys))
    feature_keys = feature_keys[:show_n_features]
    colors = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]
    time_data = data.index
    ncols = 2
    nrows = int(np.ceil(show_n_features/2))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(12, 2.5*nrows), dpi=80, facecolor="w", edgecolor="k"
    )
    for i in range(len(feature_keys)):
        key = feature_keys[i]
        c = colors[i % (len(colors))]
        t_data = data[key]
        t_data.index = time_data
        ax = t_data.plot(
            ax=axes[i // 2, i % 2],
            color=c,
            title="{}".format(key),
            rot=25,
        )
        ax.set_xlabel(None)
    plt.tight_layout()
    plt.show()
    return

def show_visualization_by_cols(data, cols):
    cols = [col for col in cols if col in data.columns]
    feature_keys = cols
    n_features = len(feature_keys)
    colors = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]
    time_data = data.index
    ncols = 2
    nrows = int(np.ceil(n_features/2))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(12, 2.5*nrows), dpi=80, facecolor="w", edgecolor="k"
    )
    for i in range(n_features):
        key = feature_keys[i]
        c = colors[i % (len(colors))]
        t_data = data[key]
        t_data.index = time_data
        if (n_features<=2):
            ax = t_data.plot(
                ax=axes[i],
                color=c,
                title="{}".format(key),
                rot=25,
            )
            ax.set_xlabel(None)
        else:
            ax = t_data.plot(
                ax=axes[i // 2, i % 2],
                color=c,
                title="{}".format(key),
                rot=25,
            )
            ax.set_xlabel(None)
    plt.tight_layout()
    plt.show()
    return

def remove_na_rows(df):
    df = df.dropna(how='all')
    return df

def get_moving_average_df(df, window):
    df = df.rolling(window=window).mean()
    return df

def set_window(df, start="2014-07-01 00:00:00", end="2014-09-01 00:00:00"):
    start = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    end = datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
    index = df.index
    index = index[index>=start]
    index = index[index<end]
    return df.loc[index]

def show_heatmap(data):
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.show()

# Data normalization

class Normalizer():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0))
        self.sd = np.std(x, axis=(0))
        normalized_x = (x - self.mu)/self.sd
        return normalized_x
    
    def transform(self, x):
        normalized_x = (x - self.mu.values)/self.sd.values
        return normalized_x        

    def inverse_transform(self, x):
        return (x*self.sd.values) + self.mu.values

def fit_normalize_df(df):
    normalizer = Normalizer()
    df = normalizer.fit_transform(df)
    return df, normalizer

def normalize_df(df, normalizer):
    df = normalizer.transform(df)
    return df

# Data loader

def concat_dataloader(dataloader, window_size):

    X_cat = []
    y_cat = []

    mid = window_size//2

    for X, y in dataloader:

        X = X.cpu().numpy()
        y = y.cpu().numpy()

        for j in range(X.shape[0]):
            X_cat.append(X[j,0,mid])
            y_cat.append(y[j,0])
    
    return X_cat, y_cat

# NILM data assessment

def analyze_appliance_signature(df, appliance, threshold):
    df = df[appliance]
    df = df.dropna()
    n = df.count()
    mx = df.max()
    avg_above = df[df>threshold].mean()

    zero_power = df[df==0].count()/n
    below_thre = df[df<=threshold].count()/n
    below_thre = below_thre - zero_power
    
    avg_below = df[df<=threshold]
    avg_below = avg_below[avg_below>0]
    avg_below = avg_below.mean()
    
    print(f"Analyze {appliance}'s energy signature...")
    print(f"Max power: {np.ceil(mx/100)*100:9.0f}W")
    print(f"@ Zero Power: {np.round(zero_power,3)*100:9.1f}% time")
    print(f"@ Below Threshold ({threshold}W): {np.round(below_thre,3)*100:3.1f}% time | {np.ceil(avg_below):5.0f}W in average")
    print(f"@ Above Threshold ({threshold}W): {np.round(1-below_thre-zero_power,3)*100:4.1f}% time | {np.ceil(avg_above/100)*100:5.0f}W in average")
    
    app_signature = np.ceil(avg_above/100)*100
    
    return app_signature

def analyze_all_appliances(df, appliances, default_threshold=40):

    defined_thresholds = { "kettle": 40, "microwave": 100, "fridge_freezer":200, "washer_dryer":1000, "dishwasher":100}

    for app in appliances:
        if app != "aggregate":
            if app in defined_thresholds.keys():
                analyze_appliance_signature(df, app, defined_thresholds[app])
            else:
                analyze_appliance_signature(df, app, default_threshold)
            print("\n")
    
    return


