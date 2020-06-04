#!/usr/bin/env python
# coding: utf-8

# In[1]:


#MOVIE LENS PROJECT ANALYSIS


# In[2]:


#1. Prepare Problem


# In[3]:


# a) Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# b) Load dataset
movie_data = pd.read_csv("movies.dat", sep="::", header=None, names=['MovieID','Title','Genres'], 
                       dtype={'MovieID': np.int32, 'Title': np.str, 'Genres': np.str}, engine='python')
users_data = pd.read_csv("users.dat", sep="::", header=None, names=['UserID','Gender','Age','Occupation','Zip-code'], 
    dtype={'UserID': np.int32, 'Gender': np.str, 'Age': np.int32, 'Occupation' : np.int32, 'Zip-code' : np.str}, engine='python')
ratings_data = pd.read_csv("ratings.dat", sep="::", header=None, names=['UserID','MovieID','Rating','Timestamp'], 
                dtype={'UserID': np.int32, 'MovieID': np.int32, 'Rating': np.int32, 'Timestamp' : np.str}, engine='python')


# In[5]:


#2. Summarize Data


# In[6]:


# Descriptive statistics on movie_data
movie_data.head()


# In[7]:


movie_data.shape


# In[8]:


movie_data.isnull().sum()
# Results show that no columns are empty or null


# In[9]:


movie_data.describe()


# In[10]:


# On users data
users_data.shape


# In[11]:


users_data.head()


# In[12]:


users_data.describe()


# In[13]:


users_data.isnull().sum()
# Results show that no columns are empty or null


# In[14]:


#Gender Distribution


# In[15]:


gender_group = users_data.groupby('Gender').size()
gender_group


# In[16]:


gender_group.plot(kind='bar', title='Gender Group Distribution')
plt.show("Male" , "Female")


# In[17]:


#The above distribution shows that most of the users are Males


# In[18]:


# On Ratings data
ratings_data.head()


# In[19]:


ratings_data.shape


# In[20]:


ratings_data.describe()


# In[21]:


ratings_data.isnull().sum()
# Results show that no columns are empty or null


# In[22]:


#User Ratings
user_group = ratings_data.groupby(['UserID']).size()
user_group.head()


# In[23]:


plt.figure(figsize=(25,10))
plt.hist(x=[ratings_data.UserID], bins=1000)
plt.show()


# In[24]:


#Merging


# In[25]:


user_ratings = pd.merge(users_data,ratings_data , how = 'inner', on = 'UserID')


# In[26]:


user_ratings = user_ratings.drop('Timestamp' , axis=1)
user_ratings = user_ratings.drop('Zip-code' , axis=1)


# In[27]:


user_ratings.head()


# In[28]:


#Now creating Master_DataSet by merging user_rating to movie_data on MovieID


# In[29]:


Master_Data = pd.merge(movie_data,user_ratings, how = 'inner', on = 'MovieID')


# In[30]:


Master_Data.shape


# In[31]:


Master_Data.head()


# In[32]:


Master_Data.describe()


# In[33]:


Master_Data.isnull().sum()
# Results show that no columns are empty or null


# In[34]:


#3. Data Visualizations


# In[35]:


#User Age Distribution


# In[36]:


age_group = Master_Data.groupby('Age').size()
age_group


# In[37]:


Master_Data.Age.plot.hist(bins=30)
plt.title('Distributions Of User-Age')
plt.show()


# In[38]:


#The above age distribution shows that most of the users are 25 years old


# In[39]:


#User rating of the movie “Toy Story”
Toy_Story_Rating = Master_Data[Master_Data.Title == "Toy Story (1995)"]


# In[40]:


plt.plot(Toy_Story_Rating.groupby("Age")["MovieID"].count(),'--bo')
Toy_Story_Rating.groupby("Age")["MovieID"].count()


# In[41]:


#The above plot shows that the Toystory movie is more popular for viewers between Age group 20-25 years


# In[42]:


#Top 25 movies by viewership rating


# In[43]:


movie_rating = Master_Data.groupby(['MovieID'], as_index=False)
average_movie_ratings = movie_rating.agg({'Rating':'mean'})
Top_Movies = Master_Data.groupby("Title").size().sort_values(ascending=False)[:25]
Top_Movies


# In[44]:


plt.ylabel("Title")
plt.xlabel("Viewership Count")
Top_Movies.plot(kind="barh")


# In[45]:


#In Top 25 Movies "American Beauty is the winner"


# In[46]:


#Rating of userid = 2696


# In[47]:


user_rating_data = Master_Data[Master_Data['UserID']==2696]
user_rating_data.head()


# In[48]:


# plotting the above data
res = user_rating_data
plt.scatter(y=res.Title, x=res.Rating)


# In[49]:


#3. Prepare Data


# In[50]:


few_viewership = Master_Data.head(500)
few_viewership.shape


# In[51]:


#preprocess data


# In[52]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(few_viewership['Age'])
x_age = le.transform(few_viewership['Age'])
x_age


# In[53]:


le.fit(few_viewership['Occupation'])
x_occ = le.transform(few_viewership['Occupation'])
x_occ


# In[54]:


le.fit(few_viewership['MovieID'])
x_movieid = le.transform(few_viewership['MovieID'])
x_movieid


# In[55]:


few_viewership['New Age'] = x_age
few_viewership['New Occupation'] = x_occ
few_viewership['New MovieID'] = x_movieid


# In[56]:


# Feature Selection
x_input = few_viewership[['New Age','New Occupation','New MovieID']]
y_target = few_viewership['Rating']


# In[57]:


x_input.head()


# In[58]:


#4. Evaluate Algorithms
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# In[59]:


# Split-out validation dataset
x_train, x_test, y_train, y_test = train_test_split(x_input, y_target, test_size=0.25)


# In[60]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[61]:


from sklearn.linear_model import LogisticRegression
logitReg = LogisticRegression()
lm = logitReg.fit(x_train, y_train)


# In[62]:


result = logitReg.predict(x_test)


# In[63]:


estimated = pd.Series(result, name='Estimated Values')


# In[64]:


final_result = pd.concat([y_test, estimated], axis=1)


# In[65]:


# Test options and evaluation metric
print (accuracy_score(y_test, result))
print (confusion_matrix(y_test, result))
print (classification_report(y_test, result))


# In[66]:


#Accuracy of the above matrix is 37.6 %


# In[67]:


# Spot-Check Algorithms
seed = 7
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[80]:


# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[81]:


#From the above plot we see that KNN gives the most accurate results


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




