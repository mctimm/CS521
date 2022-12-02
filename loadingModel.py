import pandas as pd
import numpy as np
from fastai.tabular.all import *

# Reading the datasets from excel sheet
training_set_total = pd.read_csv("../datafiles/1988.csv", skipinitialspace=True)
#test_set = pd.read_csv("../datafiles/1987.csv", skipinitialspace=True)

#create dictionary of all of the data
for file in os.listdir("../datafiles/"):
    print(file)
    print(file == '1988.csv')
    if(file == '1988.csv'):
        print("already included")
        continue
    try:
        df = pandas.read_csv("../datafiles/" + file)
    except:
        try:
            df = pandas.read_csv("../datafiles/" + file,encoding='latin-1')
        except:
            df=[]
    if(len(df) == 0):
        continue
    
    for k in df.keys():
        training_set_total[k].append(df[k])

def restructure(data):
    #data['New_Price'].fillna(0, inplace = True)

    restructured = pd.DataFrame({'DepTime': data['DepTime'],
                                'Distance': data['Distance'],
                                 'Origin': data['Origin'],
                                 'Year': data['Year'],
                                 'Month': data['Month'],
                                 'DayofMonth': data['DayofMonth'],
                                 'DayOfWeek': data['DayOfWeek'],
                                 'Dest': data['Dest'],
                                 'UniqueCarrier': data['UniqueCarrier'],
                                 'ArrDelay': data['ArrDelay']
                                 })

    return restructured


# Restructuring Training and Test sets
train_data_total = restructure(training_set_total)

#Split data  into testing vs training
train_data = train_data_total.sample(frac=0.80,random_state=1)

test_data = train_data_total.drop(train_data.index)

# this drops rows that contain an "na" from the training dataset
train_data = train_data.dropna()

print(train_data.head(5))


# Defining the keyword arguments for fastai's TabularList

# Path / default location for saving/loading models
path = '.'

# The dependent variable/target
dep_var = 'ArrDelay'

# The list of categorical features in the dataset
cat_names = ['UniqueCarrier', 'Origin', 'Dest']

# The list of continuous features in the dataset
# Exclude the Dependent variable 'Price'
cont_names = ['DepTime', 'Distance', 'Year',
              'Month', 'DayofMonth', 'DayOfWeek']

# List of Processes/transforms to be applied to the dataset
procs = [Categorify, Normalize]

# Start index for creating a validation set from train_data
start_indx = len(train_data) - int(len(train_data) * 0.2)

# End index for creating a validation set from train_data
end_indx = len(train_data)

print(train_data.dtypes)
print(test_data.dtypes)

# TabularList for Validation
#val = (TabularDataLoaders.from_df(train_data.iloc[start_indx:end_indx].copy(), path=path, cat_names=cat_names, cont_names=cont_names))

test = (TabularDataLoaders.from_df(test_data, path=path,
        cat_names=cat_names, cont_names=cont_names, procs=procs))

data = TabularDataLoaders.from_df(train_data, path=path,
                                  cat_names=cat_names, cont_names=cont_names, procs=procs, y_names=dep_var)

print(data.show_batch())

# Initializing the network
learn = tabular_learner(
    data, layers=[300, 200, 100, 50], metrics=[rmse])

learn.load("model")

#Do stuff.
