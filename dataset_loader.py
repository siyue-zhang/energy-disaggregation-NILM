
from matplotlib import rcParams
import matplotlib.pyplot as plt
from nilmtk import DataSet
from utils import *

rcParams['figure.figsize'] = (13, 6)
plt.style.use('ggplot')


def get_timeframe(building_id):

    refit = DataSet("./data/refit.h5")
    elec = refit.buildings[building_id].elec
    output = elec.get_timeframe()
    print("\n", output, "\n")
    return output

def set_ds_window(refit, start="2014-04-01", end="2014-04-10"):
    return refit.set_window(start, end)

def load_agg_app_df(refit, building_id, appliances=["dish washer", "fridge"]):
    
    elec = refit.buildings[building_id].elec
    agg_df = next(elec.mains().load())
    agg_df.columns = ["aggregate"]
    
    data = []

    for app in appliances:
        app_df = next(elec[app].load())
        app_df.columns = [app]
        data.append(app_df)
    
    df = agg_df.join(data,how="inner")
    return df

def add_hour_encoding(df):

    for i in range(24):
        df[f"hour{i}"] = df.index.hour == i
    return df




if __name__=="__main__":

    refit = DataSet("./data/refit.h5")
    df = load_agg_app_df(refit, 2, ["fridge"])
    save_file(df, "FR_df", fpath="./")