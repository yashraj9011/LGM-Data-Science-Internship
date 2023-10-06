#!/usr/bin/env python
# coding: utf-8

# In[11]:


#Import Libraries/Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


#  # Dataset Description
# The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.
# 
# Attribute Information:
# 
# 1.Sepal length in cm
# 
# 2.Sepal width in cm
# 
# 3.Petal length in cm
# 
# 4.Petal width in cm
# 
# Iris flower can be divided into 3 species as per the length and width of their Sepals and Petals:
# 
# 1.Iris Setosa
# 
# 2.Iris Versicolour
# 
# 3.Iris Virginica

# In[12]:


df=pd.read_csv("Iris.csv")
df.head()


# In[13]:


df.shape


# In[14]:


df.info


# In[15]:


df.describe


# In[16]:


#Checking Null Values
df.isnull().sum()


# In[18]:


#correlation matrix
df.corr()


# In[64]:


def meann(d):
  n = len(d)  # n is length of dataset values i.e totol obs
  mean = sum(d) / n    # mean is sum/n
  return mean

def variance(d):   
  n = len(d)
  mean = sum(d) / n
  deviations = [(x - mean) ** 2 for x in d] 
  variance = sum(deviations) / n
  return variance

def stdev(d):  # Stdev = sqroot(variance)
  import math
  var = variance(d)
  std_dev = math.sqrt(var)
  return std_dev


# In[69]:


meann(df['SepalLengthCm'])


# In[71]:


variance(df['SepalLengthCm'])


# In[73]:


stdev(df['SepalLengthCm'])


# In[86]:


def min_col(col):
    return df[col].min()
 


# In[87]:


def median_col(col):
    return df[col].median()


# In[88]:


def max_col(col):
    return df[col].max()


# In[89]:


def std_col(col):
    return df[col].std()


# In[90]:


# stats for sepalLengthCm
col = input("Enter Column name: ")
#print('Mean of column:',mean_col(col))
print('Median of column:',median_col(col))
print('Min of column:',min_col(col))
print('Max of column:',max_col(col))
print('Standard dev of column:',std_col(col))


# In[91]:


# stats for sepalWidthCm
col = input("Enter Column name: ")
#print('Mean of column:',mean_col(col))
print('Median of column:',median_col(col))
print('Min of column:',min_col(col))
print('Max of column:',max_col(col))
print('Standard dev of column:',std_col(col))


# In[92]:


# stats for PetalLengthCm
col = input("Enter Column name: ")
#print('Mean of column:',mean_col(col))
print('Median of column:',median_col(col))
print('Min of column:',min_col(col))
print('Max of column:',max_col(col))
print('Standard dev of column:',std_col(col))


# In[94]:


# stats for PetalWidthCm
col = input("Enter Column name: ")
#print('Mean of column:',mean_col(col))
print('Median of column:',median_col(col))
print('Min of column:',min_col(col))
print('Max of column:',max_col(col))
print('Standard dev of column:',std_col(col))


# In[20]:


#Correlation Heatmap
plt.figure(figsize=(9,7))
sns.heatmap(df.corr(),cmap='CMRmap',annot=True,linewidths=2)
plt.title("Correlation Graph",size=30)
plt.show()


# In[23]:


# To display no. of samples on each class.
df['Species'].unique()


# In[24]:


#Pie plot to show the overall types of Iris classifications
df['Species'].value_counts().plot(kind = 'pie',  autopct = '%1.1f%%', shadow = True, explode = [0.08,0.08,0.08])


# In[2]:


# Splitting dataset 
from sklearn.model_selection import train_test_split

features = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
X = df.loc[:, features].values   #defining the feature matrix
Y = df.Species

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 40,random_state=0)


# In[31]:


X_Train.shape


# In[32]:


X_Test.shape


# In[33]:


Y_Train.shape


# In[34]:


Y_Test.shape


# In[35]:


# Feature Scaling to bring all the variables in a single scale.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_Train = sc.fit_transform(X_Train)
X_Test = sc.transform(X_Test)


# Importing some metrics for evaluating  models.
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import  classification_report
from sklearn.metrics import confusion_matrix


# In[1]:


#Model Creation
#Logistic Regression
from sklearn.linear_model import LogisticRegression
log_model= LogisticRegression(random_state = 0)
log_model.fit(X_Train, Y_Train)

# model training
log_model.fit(X_Train, Y_Train)

# Predicting
Y_Pred_Test_log_res=log_model.predict(X_Test)


# In[37]:


Y_Pred_Test_log_res


# In[38]:


print("Accuracy:",metrics.accuracy_score(Y_Test, Y_Pred_Test_log_res)*100)


# In[39]:


print(classification_report(Y_Test, Y_Pred_Test_log_res))


# In[40]:


confusion_matrix(Y_Test,Y_Pred_Test_log_res )


# In[41]:


# Importing KNeighborsClassifier from sklearn.neighbors library
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='auto')

# Importing KNeighborsClassifier 
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

# model training
knn_model.fit(X_Train, Y_Train)

# Predicting
Y_Pred_Test_knn=knn_model.predict(X_Test)


# In[42]:


# model training
log_model.fit(X_Train, Y_Train)


# In[43]:


Y_Pred_Test_knn


# In[60]:


print("Accuracy:",metrics.accuracy_score(Y_Test,Y_Pred_Test_knn)*100)


# In[61]:


# Importing DecisionTreeClassifier from sklearn.tree library and creating an object of it  with hyper parameters criterion,splitter and max_depth.
from sklearn.tree import DecisionTreeClassifier
dec_tree = DecisionTreeClassifier(criterion='entropy',splitter='best',max_depth=6)

# model training
dec_tree.fit(X_Train, Y_Train)

# Predicting
Y_Pred_Test_dtr=dec_tree.predict(X_Test)


# In[62]:


Y_Pred_Test_dtr


# In[44]:


#Naive Bayes
from sklearn.naive_bayes import GaussianNB
nav_byes = GaussianNB()

# model training
nav_byes.fit(X_Train, Y_Train)

# Predicting
Y_Pred_Test_nvb=nav_byes.predict(X_Test)


# In[45]:


Y_Pred_Test_nvb


# In[46]:


print("Accuracy:",metrics.accuracy_score(Y_Test, Y_Pred_Test_nvb)*100)


# In[47]:


print(classification_report(Y_Test, Y_Pred_Test_nvb))


# In[48]:


confusion_matrix(Y_Test,Y_Pred_Test_nvb )


# In[49]:


#Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
Ran_for = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')

# model training
Ran_for.fit(X_Train, Y_Train)

# Predicting
Y_Pred_Test_rf=Ran_for.predict(X_Test)


# In[50]:


Y_Pred_Test_rf


# In[51]:


print("Accuracy:",metrics.accuracy_score(Y_Test,Y_Pred_Test_rf)*100)


# In[52]:


print(classification_report(Y_Test, Y_Pred_Test_rf))


# In[53]:


confusion_matrix(Y_Test,Y_Pred_Test_rf )


# In[54]:


# Importing SVC from sklearn.svm library

from sklearn.svm import SVC
svm_model=SVC(C=500, kernel='rbf')

# model training
svm_model.fit(X_Train, Y_Train)

# Predicting
Y_Pred_Test_svm=svm_model.predict(X_Test)


# In[55]:


Y_Pred_Test_svm


# In[56]:


print("Accuracy:",metrics.accuracy_score(Y_Test,Y_Pred_Test_svm)*100)


# In[57]:


print(classification_report(Y_Test, Y_Pred_Test_svm))


# In[58]:


confusion_matrix(Y_Test,Y_Pred_Test_svm )


# In[63]:


#Model Evaluation Results
print("Accuracy of Logistic Regression Model:",metrics.accuracy_score(Y_Test, Y_Pred_Test_log_res)*100)
print("Accuracy of KNN Model:",metrics.accuracy_score(Y_Test,Y_Pred_Test_knn)*100)
print("Accuracy of Decision Tree Model:",metrics.accuracy_score(Y_Test, Y_Pred_Test_dtr)*100)
print("Accuracy of Naive Bayes Model:",metrics.accuracy_score(Y_Test, Y_Pred_Test_nvb)*100)
print("Accuracy of Random Forest Classification Model:",metrics.accuracy_score(Y_Test,Y_Pred_Test_rf)*100)
print("Accuracy of SVM Model:",metrics.accuracy_score(Y_Test,Y_Pred_Test_svm)*100)


# In[ ]:





#  # Conclusion
# 
# Our dataset was not very large and consisted of only 150 rows, with all the 3 species uniformly distributed.
# PetalWidthCm was highly correlated with PetalLengthCm
# 
# PetalLengthCm was highly correlated with PetalWidthCm
# 
# I did statistical Analysis of Iris Dataset
# 
# Tried with 6 different machine learning Classification models on the Iris Test data set to classify the flower into it's three species:
# 
# a) Iris Setosa
# 
# b) Iris Versicolour
# 
# c) Iris Virginica,
# 
# based on the length and width of the flower's Petals and Sepals.​
# 
# We got very high accuracy score for all the models, and even the accuracy score of 100 for KNN and SVM with Linear Kernel models with some hyper parameter tuning maybe due to small size of dataset.List item

# In[ ]:




