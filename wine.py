# Performing PCA for wine dataset

import pandas as pd

import numpy as np

# importing the dataset

wine = pd.read_csv('C:\\Users\\WIN10\\Desktop\\LEARNING\\wine.csv')

wine.describe()

# PErforming EDA for the dataset

# Checking for NA values

wine.isna().sum()

# There are no NA values in our dataset

# Plotting histogram for getting the skewness of the data

import matplotlib.pyplot as plt

wine.columns

plt.hist(wine['Alcohol'],bins=10,color='black');plt.xlabel('Alcohol');plt.ylabel('Frequency');plt.title('Alcohol vs Frequency')

# The data is normally skewed

plt.hist(wine['Malic'],bins=10,color='black');plt.xlabel('Malic');plt.ylabel('Frequency');plt.title('Malic vs Frequency')

# The histogram is positively skewed

plt.hist(wine['Ash'],bins=10,color='black');plt.xlabel('Ash');plt.ylabel('Frequency');plt.title('Ash vs Frequency')

# Histogram is normally skewed

plt.hist(wine['Alcalinity'],bins=10,color='black');plt.xlabel('Alcalinity');plt.ylabel('Frequency');plt.title('Alcalinity vs Frequency')

# Histogram is normally skewed

plt.hist(wine['Magnesium'],bins=10,color='black');plt.xlabel('Magnesium');plt.ylabel('Frequency');plt.title('Magnesium vs Frequency')

# Histogram is positively skewed

plt.hist(wine['Phenols'],bins=10,color='black');plt.xlabel('Phenols');plt.ylabel('Frequency');plt.title('Phenols vs Frequency')

# Histogram is normally skewed

plt.hist(wine['Flavanoids'],bins=10,color='black');plt.xlabel('Flavanoids');plt.ylabel('Frequency');plt.title('Flavanoids vs Frequency')

# Histogram is normally skewed

plt.hist(wine['Nonflavanoids'],bins=10,color='black');plt.xlabel('Nonflavanoids');plt.ylabel('Frequency');plt.title('Nonflavanoids vs Frequency')

# Histogram is normally skewed

plt.hist(wine['Proanthocyanins'],bins=10,color='black');plt.xlabel('Proanthocyanins');plt.ylabel('Frequency');plt.title('Proanthocyanins vs Frequency')

# Histogram is positively skewed

plt.hist(wine['Color'],bins=10,color='black');plt.xlabel('Color');plt.ylabel('Frequency');plt.title('Color vs Frequency')

# Histogram is positively skewed

plt.hist(wine['Hue'],bins=10,color='black');plt.xlabel('Hue');plt.ylabel('Frequency');plt.title('hue vs Frequency')

# Histogram is normally skewed

plt.hist(wine['Dilution'],bins=10,color='black');plt.xlabel('Dilution');plt.ylabel('Frequency');plt.title('Dilution vs Frequency')

# Histogram is negatively skewed

plt.hist(wine['Proline'],bins=10,color='black');plt.xlabel('Proline');plt.ylabel('Frequency');plt.title('Proline vs Frequency')

# Histogram is positively skewed

# Plotting boxplot for identifying outliers

plt.boxplot(wine['Alcohol'],vert='False');plt.xlabel('Alcohol');plt.ylabel('Frequency');plt.title('Alcohol vs Frequency')

# There are no outliers in this column

plt.boxplot(wine['Malic'],vert='False');plt.xlabel('Malic');plt.ylabel('Frequency');plt.title('Malic vs Frequency')

# outliers are present in this column

# Therefore we apply winsorization method and replace

IQR1 = wine['Malic'].quantile(0.75)-wine['Malic'].quantile(0.25)

lowerlimit1 = wine['Malic'].quantile(0.25) - (1.5*IQR1)

lowerlimit1

upperlimit1 = wine['Malic'].quantile(0.75) + (1.5*IQR1)

upperlimit1

wins1a = wine['Malic'].quantile(0.05)

wins1a

wins1b = wine['Malic'].quantile(0.95)

wins1b

# Replacing the outliers

wine['Malic'] = pd.DataFrame(np.where(wine['Malic']<lowerlimit1,wins1a,np.where(wine['Malic']>upperlimit1,wins1b,wine['Malic'])))

plt.boxplot(wine['Malic'])

# The outliers are replaced

plt.boxplot(wine['Ash'],vert='False');plt.xlabel('ASh');plt.ylabel('Frequency');plt.title('Ash vs Frequency')

# Outliers are present in this column

IQR2 = wine['Ash'].quantile(0.75) - wine['Ash'].quantile(0.25)

lowerlimit2 = wine['Ash'].quantile(0.25)-(IQR2*1.5)

lowerlimit2

upperlimit2 = wine['Ash'].quantile(0.75)+(IQR2*1.5)

upperlimit2 

wins2a = wine['Ash'].quantile(0.05)

wins2a

wins2b = wine['Ash'].quantile(0.95)

wins2b

# Replacing the outliers

wine['Ash'] = pd.DataFrame(np.where(wine['Ash']<lowerlimit2,wins2a,np.where(wine['Ash']>upperlimit2,wins2b,wine['Ash'])))

plt.boxplot(wine['Ash']) # Outliers are replaced

plt.boxplot(wine['Alcalinity']);plt.xlabel('Alcalinity');plt.ylabel('frequency');plt.title('Alcalintiy vs Frequency')

# outliers are present in this column

# Therefore we replace the outliers by winsorization method

IQR3 = wine['Alcalinity'].quantile(0.75)-wine['Alcalinity'].quantile(0.25)

lowerlimit3 = wine['Alcalinity'].quantile(0.25) - (1.5*IQR3)

lowerlimit3

upperlimit3 = wine['Alcalinity'].quantile(0.75)+(1.5*IQR3)

upperlimit3

wins3a = wine['Alcalinity'].quantile(0.05)

wins3b = wine['Alcalinity'].quantile(0.95)

# Replacing the outliers

wine['Alcalinity'] = pd.DataFrame(np.where(wine['Alcalinity']<lowerlimit3,wins3a,np.where(wine['Alcalinity']>upperlimit3,wins3b,wine['Alcalinity'])))

plt.boxplot(wine['Alcalinity'])  # Outliers are replaced

plt.boxplot(wine['Magnesium']);plt.xlabel('Magnesium');plt.ylabel('frequency');plt.title('Magnesium vs Frequency')

# Outliers are present in this column

# Therefore we replace the outliers by winsorization method

IQR4 = wine['Magnesium'].quantile(0.75)-wine['Magnesium'].quantile(0.25)

lowerlimit4 = wine['Magnesium'].quantile(0.25)-(1.5*IQR4)

upperlimit4 = wine['Magnesium'].quantile(0.75)+(1.5*IQR4)

wins4a = wine['Magnesium'].quantile(0.05)

wins4b = wine['Magnesium'].quantile(0.95)

# Replacing the outliers

wine['Magnesium']=pd.DataFrame(np.where(wine['Magnesium']<lowerlimit4,wins4a,np.where(wine['Magnesium']>upperlimit4,wins4b,wine['Magnesium'])))

plt.boxplot(wine['Magnesium']) # Outliers are reaplaced

plt.boxplot(wine['Phenols']);plt.xlabel('Phenols');plt.ylabel('frequency');plt.title('Phenols vs Frequency')

# There are no outliers in this column

plt.boxplot(wine['Flavanoids']);plt.xlabel('Flavanoids');plt.ylabel('frequency');plt.title('Flavanoids vs Frequency')

# There are no outliers in this column

plt.boxplot(wine['Nonflavanoids']);plt.xlabel('Nonflavanoids');plt.ylabel('frequency');plt.title('Nonflavanoids vs Frequency')

# There are no outliers in this column

plt.boxplot(wine['Proanthocyanins']);plt.xlabel('Proanthocyanins');plt.ylabel('frequency');plt.title('Proanthocyanins vs Frequency')

# Outliers are present in this column

IQR5 = wine['Proanthocyanins'].quantile(0.75)-wine['Proanthocyanins'].quantile(0.25)

lowerlimit5 = wine['Proanthocyanins'].quantile(0.25)-(1.5*IQR5)

upperlimit5 = wine['Proanthocyanins'].quantile(0.75)+(1.5*IQR5)

wins5a = wine['Proanthocyanins'].quantile(0.05)

wins5b = wine['Proanthocyanins'].quantile(0.95)

# Replacing the outliers

wine['Proanthocyanins'] = pd.DataFrame(np.where(wine['Proanthocyanins']<lowerlimit5,wins5a,np.where(wine['Proanthocyanins']>upperlimit5,wins5b,wine['Proanthocyanins'])))

plt.boxplot(wine['Proanthocyanins']) # Outliers are replaced

plt.boxplot(wine['Color']);plt.xlabel('Color');plt.ylabel('frequency');plt.title('Color vs Frequency')

# Outliers are present in this column

IQR6 = wine['Color'].quantile(0.75)-wine['Color'].quantile(0.25)

lowerlimit6=wine['Color'].quantile(0.25)-(1.5*IQR6)

upperlimit6 = wine['Color'].quantile(0.75) + (1.5*IQR6)

wins6a = wine['Color'].quantile(0.05)

wins6b = wine['Color'].quantile(0.95)

wine['Color'] = pd.DataFrame(np.where(wine['Color']<lowerlimit6,wins6a,np.where(wine['Color']>upperlimit6,wins6b,wine['Color'])))

plt.boxplot(wine['Color'])  # Outliers are replaced

plt.boxplot(wine['Hue']);plt.xlabel('Hue');plt.ylabel('frequency');plt.title('Hue vs Frequency')

# There are outliers in this column

# Therefore we will replace them by winsorization method

IQR7 = wine['Hue'].quantile(0.75) - wine['Hue'].quantile(0.25)

lowerlimit7 = wine['Hue'].quantile(0.25) - (1.5*IQR7)

upperlimit7 = wine['Hue'].quantile(0.75) + (1.5*IQR7)

wins7a = wine['Hue'].quantile(0.05)

wins7b = wine['Hue'].quantile(0.95)

wine['Hue'] = pd.DataFrame(np.where(wine['Hue']<lowerlimit7,wins7a,np.where(wine['Hue']>upperlimit7,wins7b,wine['Hue'])))

plt.boxplot(wine['Hue'])  # Outliers are replaced

plt.boxplot(wine['Dilution']);plt.xlabel('Dilution');plt.ylabel('frequency');plt.title('Dilution vs Frequency')

# There are no outliers in this column

plt.boxplot(wine['Proline']);plt.xlabel('Proline');plt.ylabel('frequency');plt.title('Proline vs Frequency')

# There are no outliers in this column

# Checking the normality of data

import scipy.stats as stats

import pylab

stats.probplot(wine['Alcohol'],dist='norm',plot=pylab)

# The data is normal

stats.probplot(wine['Malic'],dist='norm',plot=pylab)

# The data is non normal, therefore we apply transformation

stats.probplot(np.log(wine['Malic']),dist='norm',plot=pylab)

stats.probplot(np.sqrt(wine['Malic']),dist='norm',plot=pylab)

stats.probplot((1/wine['Malic']),dist='norm',plot=pylab)

# The data is non normal even after applying the transformations

stats.probplot(wine['Ash'],dist='norm',plot=pylab)

# The data is normal

stats.probplot(wine['Alcalinity'],dist='norm',plot=pylab)

# The data is normal

stats.probplot(wine['Magnesium'],dist='norm',plot=pylab)

# The data is normal

stats.probplot(wine['Phenols'],dist='norm',plot=pylab)

# The data is normal

stats.probplot(wine['Flavanoids'],dist='norm',plot=pylab)

# The data is normal

stats.probplot(wine['Nonflavanoids'],dist='norm',plot=pylab)

# The data is normal

stats.probplot(wine['Proanthocyanins'],dist='norm',plot=pylab)

# The data is normal

stats.probplot(wine['Color'],dist='norm',plot=pylab)

# The data is normal

stats.probplot(wine['Hue'],dist='norm',plot=pylab)

# The data is normal

stats.probplot(wine['Dilution'],dist='norm',plot=pylab)

# The data is non normal, therefore we apply transformation

stats.probplot(np.log(wine['Dilution']),dist='norm',plot=pylab)

stats.probplot(np.sqrt(wine['Dilution']),dist='norm',plot=pylab)

stats.probplot((1/wine['Dilution']),dist='norm',plot=pylab)

# Even after applying the transformation the data is non normal

stats.probplot(wine['Proline'],dist='norm',plot=pylab)

# The data is non normal

stats.probplot(np.log(wine['Proline']),dist='norm',plot=pylab)

# The data is normal

wine['Proline'] = np.log(wine['Proline'])

# Now we check for variance in all columns

print(np.var(wine,axis=0))

# We have variance in all columns

# Now performing PCA

from sklearn.decomposition import PCA

from sklearn.preprocessing import scale

# Dropping the Type column

wine1 = wine.drop(['Type'],axis=1)

# Normalizing the data

normdata = scale(wine1)

pca = PCA(n_components=6)

pca_values = pca.fit_transform(normdata)

# The amount of variance explains the importance of each component 

var = pca.explained_variance_ratio_

var

pca.components_

pca.components_[0]

# Cumulative varicance

var1 = np.cumsum(np.round(var,decimals=4)*100)

var1

# Varicance plot for PCA components

plt.plot(var1,color='blue')

# PCA scores

pca_values

pca_data = pd.DataFrame(pca_values)

pca_data.columns = 'comp0','comp1','comp2','comp3','comp4','comp5'

hclustfinal = pd.concat([wine.Type,pca_data.iloc[:,0:3]],axis=1)

# Scatter Diagram

plt.scatter(x=hclustfinal.comp0,y=hclustfinal.comp1)



# Performing clustering 

# Plotting Dendrogram

from scipy.cluster.hierarchy import linkage

import scipy.cluster.hierarchy as sch

z = linkage(hclustfinal,metric='euclidean',method='complete')

plt.figure(figsize=(15,8));plt.title('Hieraricha Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')

sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)

# Doing Agglomerative Clustering

from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(hclustfinal)

h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

wine1['Clust'] = cluster_labels   # Creating 'Clust' Column

wine = wine1.iloc[:,[13,0,1,2,3,4,5,6,7,8,9,10,11,12]]

wine.head()

# Finding the aggregate

wine1.iloc[:,:].groupby(wine.Clust).mean()

# Creating a csv file 

wine.to_csv("hclusterwine.csv", encoding = "utf-8")

import os

os.getcwd()



# Performing K Means

# Plotting elbow curve to decide K value

from sklearn.cluster import KMeans

twss = []

k = list(range(2,9))  # Creating a range of values from 2-8

# Writing a user defined function for KMeans

for i in k:

    Kmeans = KMeans(n_clusters=i)

    Kmeans.fit(hclustfinal)

    twss.append(Kmeans.inertia_)

twss

plt.plot(k,twss,'ro-');plt.xlabel('No of Clusters');plt.ylabel('Total Within SS')

# Selecting Cluster

# From scree plot we will select 3 as number of clusters

model = KMeans(n_clusters=3) 

model.fit(hclustfinal)

model.labels_

mb = pd.Series(model.labels_)

kmeansclust = pd.concat([mb,wine1.iloc[:,0:13]],axis=1)

columns2 = columns2=([                'K_cluster',         'Alcohol',           'Malic',

                   'Ash',      'Alcalinity',       'Magnesium',

               'Phenols',      'Flavanoids',   'Nonflavanoids',

       'Proanthocyanins',           'Color',             'Hue',

              'Dilution',         'Proline'])

kmeansclust.columns = columns2

kmeansclust.head()

# Finding the Aggregate

wine1.iloc[:,:].groupby(kmeansclust.K_cluster).mean()

# creating a csv file 

kmeansclust.to_csv("kmeanscluster.csv", encoding = "utf-8")

import os

os.getcwd()



# From both Hierarchical and K-Means clustering we can conclude that, 

# the wine belonging to group 1 have high alcohol percentage and color,

# So, may be this group maybe classified as Premium Quality Wines 



# The wine belonging to group 2 have moderate alcohol percentage, ash content and color,

# therefore, this group may be classified as White Wine section 



# The wine belonging to group 3 have less alcohol percentage, high ash content and color

# therefore, this group maybe classified as Low Quality Wines 



