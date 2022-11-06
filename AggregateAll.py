from numpy import unique
from numpy import where
import numpy as np
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN
from matplotlib import pyplot
import pandas;
import math
import os


arrDelays = dict()

depDelays = dict()

airPoints = list()
df = pandas.read_csv("averageDelayData.csv")
print(df.keys())


airports = df["airport"].apply(lambda x: x[0:len(x)-4]).unique()
print(airports)
for airport in airports:
    arr = df[df["airport"].str.contains(airport)]
    arr['arr_combined_mean_count'] = arr['arrDelay']*arr['arrCount']
    arr['dep_combined_mean_count'] = arr['depDelay']*arr['depCount']
    if(arr['depCount'].sum() == 0):
        depDelays[airport] = 0
    else:
        depDelays[airport] = arr['dep_combined_mean_count'].sum()/arr['depCount'].sum()
    if(arr['arrCount'].sum() == 0):
        arrDelays[airport] = 0
    else:
        arrDelays[airport] = arr['arr_combined_mean_count'].sum()/arr['arrCount'].sum()
    airPoints.append([arrDelays[airport],depDelays[airport]])
#print(airPoints)
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

