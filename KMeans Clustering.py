#!/usr/bin/env python
# coding: utf-8

# # IMPORTING LIBRARIES & DATASET

# In this step we import all libraires that is required to proces our datset. Following this we will import our dataset. And the dataset that we use is the Happiness Index dataset

# In[1]:


#Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.cluster as cluster
import sklearn.metrics as metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[2]:


#Importing Dataset
dataset = pd.read_csv("happiness_data.csv")
dataset


# # MISSING VALUES

# In this stage we will check if there is any missing values that are present in our dataset. this step is essential as presence of mising values could be serious bottlenecks in the processing of our dataset

# In[3]:


#Finding Missing Values
print(dataset.isnull().sum())


# # DATATYPE

# The following step helps to identify the dtatype present in our dataset. 

# In[4]:


#Knowing the datatypes
dataset.dtypes


# # CORRELATION

# In this step we analyse the correlation between features . this step is essential as it helps to identify the redundant features in our dataset

# In[5]:


#Finding the correlation between features using Heatmaps
plt.figure(figsize=(10,8))
sns.heatmap(dataset.corr(), annot = True)


# # REMOVING REDUNDANT FEATURES

# In this step,we filter out all the redundant features that are not necessary for our analysis. In the below step we take all those features that have correlation above 95%. 

# In[6]:


#Finding the features to be dropped
def correlation(dataset,threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                col_name = corr_matrix.columns[i]
                col_corr.add(col_name)
    return col_corr

#Numeric feature
num_features = dataset.select_dtypes(include=[np.number])
x= num_features

#List of Features to be dropped
corr_features = correlation(x, 0.95)
corr_features


# In[8]:


#List of Features to be dropped
corr_features = correlation(x, 0.95)
len(set(corr_features))


# In[9]:


corr_features


# In[12]:


df_unclean = dataset.drop(labels=['Explained by: Freedom to make life choices',
 'Explained by: Generosity',
 'Explained by: Healthy life expectancy',
 'Explained by: Log GDP per capita',
 'Explained by: Perceptions of corruption',
 'Explained by: Social support',
 'lowerwhisker',
 'upperwhisker'],axis=1)
df_unclean


# # KNOWING OUR DATASET

# In[13]:


df_unclean.describe()


# In[14]:


df_unclean.shape


# In[15]:


#Removing Outliers
df_unclean.dtypes


# # DROPPING STRINGS

# Since strings cannot be normalized, we will rmove all the strings in our dataset, before normalizing the data

# In[17]:


# Removing strings
df = df_unclean.drop(labels=['Country name','Regional indicator'],axis=1)
df.shape


# # NORMALIZING THE DATASET 

# Data Normalization is the process of rescaling the data in a dataset and improving its integrity 
# by eliminating data redundancy. 

# In[18]:


#Normalize the data attributes 
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
names = df.columns
d = scaler.fit_transform(df)
df_scaled = pd.DataFrame(d, columns=names)
df_scaled.head()


# # DIMENSIONALITY REDUCTION

# Dimensionality Reduction is the process by which we convert a high dimensionality dataset
# into a low dimensionality dataset. In this step we aim to reduce our dataset into a lower
# dimensional dataset that would capture a variance above 90%. 

# In[19]:


# Plotting graph to find the ideal number of n_components
pca = PCA()
principalcomponents = pca.fit_transform(df_scaled)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')
plt.title('Explained variance')
plt.show


# The above graph proves that variance above 90% is captured when number of components is
# equal to 4.  So, we take the number of components as 4: 

# In[23]:


# Create a PCA instance: pca
pca = PCA(n_components= 4)
principalcomponents = pca.fit_transform(df_scaled)


# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='blue')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)


# In[24]:


principalcomponents.shape


# In[25]:


final_data = pd.DataFrame(principalcomponents)
final_data


# # IDEAL NUMBER OF CLUSTERS

# There are several methods to identify the ideal number of clusters like the yellow brick method, 
# Elbow method, Inter cluster distance method, Silhouette score method, etc. For our analysis,
# we are taking Elbow method and Silhouette score methods to identify the ideal number of
# clusters.

# # ELBOW METHOD

# Elbow method is a method through which we can identify the ideal number of clusters. In this 
# method we plot WCSS (Within-cluster Sum of Squares) and number of clusters in a graph

# In[27]:


#Determining number of clusters using Elbow method
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
   model = KMeans(n_clusters = i, init = "k-means++")
   model.fit(final_data.iloc[:,:2])
   wcss.append(model.inertia_)
plt.figure(figsize=(5,5))
plt.plot(range(1,11), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# # SILHOUETTE SCORE METHOD

# In[ ]:


As mentioned before Silhouette score method is method that is used to identify the ideal 
number of clusters. In this method we randomly assign certain number of clusters to calculate
the silhouette score corresponding to each cluster. The ideal cluster is the cluster having the
highest silhouette score. 


# In[28]:


#Determining number of clusters using Silhouette score method
import sklearn.cluster as cluster
import sklearn.metrics as metrics

for i in range (2,7):
    labels=cluster.KMeans(n_clusters = i, init="k-means++", random_state=200).fit(final_data).labels_
    print("Silhouette score for k(clusters) = "+str(i)+" is"
         +str(metrics.silhouette_score(final_data, labels,metric="euclidean", sample_size=1000, random_state=200)))


# As we run this code, we will get the silhouette score for each cluster ranging from 2 to 6. From 
# the table below we can understand that cluster number 2 is having the highest
# silhouette score which is 0.374.  Hence from these two methods we can conclude that the ideal number of clusters is two.

# # CLUSTERING 

# After reducing the dimensionality and finding the ideal number of clusters, now our data set is 
# ready for performing K-Means.

# In[29]:


# Clustering using KMeans
import numpy as np
model = KMeans(n_clusters = 2, init = "k-means++")
label = model.fit_predict(final_data.iloc[:,:2])
centers = np.array(model.cluster_centers_)
uniq = np.unique(label)

# colors for plotting
colors = ['red', 'green']
# assign a color to each features (note that we are using features as target)
features_colors = [ colors[label[i]] for i in range(len(final_data.iloc[:,:2])) ]
T=final_data.iloc[:,:2]

# plot the PCA cluster components
plt.scatter(T[0], T[1],
            c=features_colors, marker='o',
            alpha=0.4)


# In[30]:


# plot the centroids
plt.scatter(centers[:, 0], centers[:, 1],
            marker='x', s=100,
            linewidths=3, c=colors
        )

# store the values of PCA component in variable: for easy writing
xvector =  pca.components_[0] * max(T[0])
yvector =  pca.components_[1] * max(T[1])
columns = df.columns


# In[31]:


# plot the 'name of individual features' along with vector length
for i in range(len(columns)):
    # plot arrows
    plt.arrow(0, 0, xvector[i], yvector[i],
                color='b', width=0.005,
                head_width=0.08, alpha=0.5
            )
    # plot name of features
    plt.text(xvector[i], yvector[i], list(columns)[i], color='b', alpha=0.75)

plt.scatter(T[0], T[1], 
            c=features_colors, marker='o',
            alpha=0.4)

#plot the centroids
plt.scatter(centers[:, 0], centers[:, 1],
            marker='x', s=100,
            linewidths=3, c=colors )            
plt.show()


# # ANALYSIS

# Now we got ourdataset divided into two - Cluster 1 and Cluster 2. In one cluster the Perception of corruption and standard error of ladder score is high which we refer as Dystopia countries(58). And the rest belong to Utopia countries(91) 

# In[32]:


label


# In[33]:


df_clustered=df_unclean
df_clustered['clusters']=label.tolist()
df_clustered


# In[34]:


#Dystopia countries
df_clustered[df_clustered.clusters==1]


# In[35]:


df_clustered[df_clustered.clusters==1].shape


# In[36]:


#Utopian Countries
df_clustered[df_clustered.clusters==0]


# In[38]:


df_clustered.sort_values( by="Ladder score",ascending=False)


# In[39]:


df_clustered.sort_values( by="Logged GDP per capita",ascending=False)


# In[40]:


df_clustered.sort_values( by="Social support",ascending=False)


# In[41]:


df_clustered.sort_values( by="Healthy life expectancy",ascending=False)


# In[42]:


df_clustered.sort_values( by="Freedom to make life choices",ascending=False)


# In[43]:


df_clustered.sort_values( by="Generosity",ascending=False)


# In[44]:


df_clustered.sort_values( by="Perceptions ofcorruption",ascending=False)


# In[ ]:




