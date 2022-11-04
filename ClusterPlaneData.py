from numpy import unique
from numpy import where
import numpy as np
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN
from matplotlib import pyplot
import pandas;
import math
import os

#creating the dataset
arrDelays = dict()

depDelays = dict()
airPoints = list()
depCounts = dict()
arrCounts = dict()

for file in os.listdir("../datafiles/"):
    try:
        df = pandas.read_csv("../datafiles/" + file)
    except:
        try:
            df = pandas.read_csv("../datafiles/" + file,encoding='latin-1')
        except:
            df=[]
    if(len(df) == 0):
        continue
    #split by airport

    airports = np.unique(np.append(df["Dest"].unique(),df["Origin"].unique()))
    airports = map(lambda x: (x + file.split(".")[0],x),airports)
    for airport in airports:
        arr = df[df["Dest"] == airport[1]]
        if(len(arr) > 0):
            arrDelays[airport[0]] = arr["ArrDelay"].dropna().mean()
            arrCounts[airport[0]] = arr["ArrDelay"].dropna().count()
        else:
            arrDelays[airport[0]] = 0
            arrCounts[airport[0]] = 0 
        dep =  df[df["Origin"] == airport[1]]
        if(len(dep) > 0):
            depDelays[airport[0]] =dep["DepDelay"].dropna().mean()
            depCounts[airport[0]] =dep["DepDelay"].dropna().count()
        else: 
            depDelays[airport[0]] = 0
            depCounts[airport[0]] = 0
        if(math.isnan(depDelays[airport[0]])):
            depDelays[airport[0]] = 0
            depCounts[airport[0]] = 0
        if(math.isnan(arrDelays[airport[0]])):
            arrDelays[airport[0]] = 0
            arrCounts[airport[0]] = 0 
        airPoints.append([arrDelays[airport[0]],depDelays[airport[0]]])
import csv
dicts = (depDelays, depCounts, arrDelays,arrCounts)

with open('averageDelayData.csv', 'w',newline='') as ofile:
    writer = csv.writer(ofile, delimiter=' ')
    writer.writerow(['airport', 'depDelays', 'depCount', 'arrDelay', 'arrCount'])
    for key in depCounts.keys():
        writer.writerow([key] + [d[key] for d in dicts])

#use DBSCAN
X = np.array(airPoints)
# define the model
model = DBSCAN(eps=0.75, min_samples=5)
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
fig, ax = pyplot.subplots()
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	ax.scatter(X[row_ix, 0], X[row_ix, 1])
#annotate
#for key in depDelays.keys():
#    ax.annotate(key,(arrDelays[key],depDelays[key]))

ax.set_ylabel("departure delays")
ax.set_xlabel("arrival delays")
# show the plot
pyplot.show()

