# Problem on PCA for heart diesease data

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

# Importing the dataset

heartdisease = pd.read_csv('C:\\Users\\WIN10\\Desktop\\LEARNING\\heart disease.csv')

heartdisease.describe()

# Performing EDA for the dataset

# Checking for NA values in our dataset

heartdisease.isna().sum()

# There are no NA values in out dataset

# Plotting histogram for identifying the skewness 

plt.hist(heartdisease['trestbps'],bins=10,color='blue');plt.xlabel('trestbps');plt.ylabel('Frequency');plt.title('trestbps vs Frequency')

# The histogram is positively skewed

plt.hist(heartdisease['chol'],bins=10,color='blue');plt.xlabel('chol');plt.ylabel('Frequency');plt.title('chol vs Frequency')

# The histogram is positively skewed

plt.hist(heartdisease['thalach'],bins=10,color='blue');plt.xlabel('thalach');plt.ylabel('Frequency');plt.title('thalach vs Frequency')

# The histogram is negatively skewed

plt.hist(heartdisease['oldpeak'],bins=10,color='blue');plt.xlabel('oldpeak');plt.ylabel('Frequency');plt.title('oldpeak vs Frequency')

# The histogram is positively skewed

# Plotting boxplot for identifying outliers

plt.boxplot(heartdisease['trestbps']);plt.xlabel('trestbps');plt.ylabel('Frequency');plt.title('trestbps vs Frequency')

# Outliers are present in this column

# Therefore we will do winsorization method 

IQR1 = heartdisease['trestbps'].quantile(0.75) - heartdisease['trestbps'].quantile(0.25)

lowerlimit1 = heartdisease['trestbps'].quantile(0.25) - (1.5*IQR1)

upperlimit1 = heartdisease['trestbps'].quantile(0.75) + (1.5*IQR1)

wins1a = heartdisease['trestbps'].quantile(0.05)

wins1b = heartdisease['trestbps'].quantile(0.95)

# Replacing the outliers 

heartdisease['trestbps'] = pd.DataFrame(np.where(heartdisease['trestbps']<lowerlimit1,wins1a,np.where(heartdisease['trestbps']>upperlimit1,wins1b,heartdisease['trestbps'])))

plt.boxplot(heartdisease['trestbps'])

# Outliers are replaced

plt.boxplot(heartdisease['chol']);plt.xlabel('chol');plt.ylabel('Frequency');plt.title('chol vs Frequency')

# Outliers are present in this column

IQR2 = heartdisease['chol'].quantile(0.75) - heartdisease['chol'].quantile(0.25)

lowerlimit2 = heartdisease['chol'].quantile(0.25) - (1.5*IQR2)

upperlimit2 = heartdisease['chol'].quantile(0.75) + (1.5*IQR2)

wins2a = heartdisease['chol'].quantile(0.05)

wins2b = heartdisease['chol'].quantile(0.95)

# Replacing the outliers

heartdisease['chol'] = pd.DataFrame(np.where(heartdisease['chol']<lowerlimit2,wins2a,np.where(heartdisease['chol']>upperlimit2,wins2b,heartdisease['chol'])))

plt.boxplot(heartdisease['chol'])

# Outliers are replaced

plt.boxplot(heartdisease['thalach']);plt.xlabel('thalach');plt.ylabel('Frequency');plt.title('thalach vs Frequency')

# Outlieres are present in this column

IQR3 = heartdisease['thalach'].quantile(0.75) - heartdisease['thalach'].quantile(0.25)

lowerlimit3 = heartdisease['thalach'].quantile(0.25) - (IQR3*1.5)

upperlimit3 = heartdisease['thalach'].quantile(0.75) + (1.5*IQR3)

wins3a = heartdisease['thalach'].quantile(0.05)

wins3b = heartdisease['thalach'].quantile(0.95)

heartdisease['thalach'] = pd.DataFrame(np.where(heartdisease['thalach']<lowerlimit3,wins3a,np.where(heartdisease['thalach']>upperlimit3,wins3b,heartdisease['thalach'])))

plt.boxplot(heartdisease['thalach'])

# The outliers are replaced

plt.boxplot(heartdisease['oldpeak']);plt.xlabel('oldpeak');plt.ylabel('Frequency');plt.title('oldpeak vs Frequency')

# Outliers are present in this column

IQR4 = heartdisease['oldpeak'].quantile(0.75) - heartdisease['oldpeak'].quantile(0.25)

lowerlimit4 = heartdisease['oldpeak'].quantile(0.25) - (1.5*IQR4)

upperlimit4 = heartdisease['oldpeak'].quantile(0.75) + (1.5*IQR4)

wins4a = heartdisease['oldpeak'].quantile(0.05)

wins4b = heartdisease['oldpeak'].quantile(0.95)

# Replacing the outliers

heartdisease['oldpeak'] = pd.DataFrame(np.where(heartdisease['oldpeak']<lowerlimit4,wins4a,np.where(heartdisease['oldpeak']>upperlimit4,wins4b,heartdisease['oldpeak'])))

plt.boxplot(heartdisease['oldpeak'])

# Outliers are replaced

# Now checking the normality of data

import scipy.stats as stats

import pylab

stats.probplot(heartdisease['trestbps'],dist='norm',plot=pylab)

# The data is normal

stats.probplot(heartdisease['chol'],dist='norm',plot=pylab)

# THe data is normal

stats.probplot(heartdisease['thalach'],dist='norm',plot=pylab)

# The data is normal

stats.probplot(heartdisease['oldpeak'],dist='norm',plot=pylab)

# The data is non normal, therefore we apply transformations

stats.probplot(np.log(heartdisease['trestbps']),dist='norm',plot=pylab)

stats.probplot(np.sqrt(heartdisease['trestbps']),dist='norm',plot=pylab)

# The data is normal after applying sqrt transformation

heartdisease['oldpeak'] = np.sqrt(heartdisease['oldpeak'])

# Checking for the variance in all columns

print(np.var(heartdisease,axis=0))

# We have variance in all the column

# Performing PCA on the dataset

from sklearn.decomposition import PCA

from sklearn.preprocessing import scale

# Normalising the data

normdata = scale(heartdisease)

heartpca = PCA(n_components=6)

heartpcavalues = heartpca.fit_transform(normdata)

# The amount of variance explains the importance of each component

var = heartpca.explained_variance_ratio_

var

heartpca.components_

heartpca.components_[0]

# Cumulative Variance

var1 = np.cumsum(np.round(var,decimals=4)*100)

var1

# Variance plot for PCA components obtained

plt.plot(var1,color='black')

# PCA values

heartpcavalues

pca = pd.DataFrame(heartpcavalues)

pca.columns = 'comp0','comp1','comp2','comp3','comp4','comp5'

pca = pca.drop(['comp3','comp4','comp5'],axis=1)

# Plotting a scatter diagram

plt.scatter(x=pca.comp0,y=pca.comp1)



# PErforming Hiererchial Clustering

from scipy.cluster.hierarchy import linkage

import scipy.cluster.hierarchy as sch

# Distance matrix

z = linkage(pca,method='complete',metric='euclidean')

# Plotting Dendrogram

plt.figure(figsize=(15,8));plt.title('Hierarchial Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')

sch.dendrogram(z,leaf_font_size=10,leaf_rotation=0)

# Now applying Agglomerative clustering 

from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(pca)

h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

hcluster = pd.concat([cluster_labels,heartdisease.iloc[:,0:14]],axis=1)

hcluster.columns

columns1=([                'cluster',        'age',      'sex',       'cp', 'trestbps',     'chol',

            'fbs',  'restecg',  'thalach',    'exang',  'oldpeak',    'slope',

             'ca',     'thal',   'target'])

hcluster.columns = columns1

hcluster.head()

# Finding aggregate

heartdisease.iloc[:,:].groupby(hcluster.cluster).mean()

# creating a csv file 

hcluster.to_csv("hclustheartdata.csv", encoding = "utf-8")

import os

os.getcwd()



# Performing K Means

# Plotting elbow for deciding the K value

from sklearn.cluster import KMeans

twss = []

k = list(range(2,9)) # Creating a range of values from 2-8

# Defining a user defined function for KMeans

for i in k:

    kmeans = KMeans(n_clusters=i)

    kmeans.fit(pca)

    twss.append(kmeans.inertia_)

twss

plt.plot(k,twss,'ro-');plt.xlabel('No of clusters');plt.ylabel('total within SS')

# Selecting cluster

model = KMeans(n_clusters=3)

model.fit(pca)

model.labels_

mb = pd.Series(model.labels_)

kmeansclust = pd.concat([mb,heartdisease.iloc[:,0:14]],axis=1)

kmeansclust.columns

columns2=([                'K_cluster',         'age',      'sex',       'cp', 'trestbps',     'chol',

            'fbs',  'restecg',  'thalach',    'exang',  'oldpeak',    'slope',

             'ca',     'thal',   'target'])

kmeansclust.columns = columns2

kmeansclust.head()

# Finding aggregate mean of each cluster

heartdisease.iloc[:,:].groupby(kmeansclust.K_cluster).mean()

# creating a csv file 

kmeansclust.to_csv("kmeansclusteheart.csv", encoding = "utf-8")

import os

os.getcwd()



# From both Hierarchial and K-Means clustering we can conclude that,

# The person belonging to high age group has high cholesterol ratio,

# so the pharmaceutical company should focus on lower age groups for implementing drugs there,

# because, the older the people become the higher the attack ratio and risky to implement on them

