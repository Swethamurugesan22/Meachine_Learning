#!/usr/bin/env python
# coding: utf-8

# <h2>Titanic Passanger Survival Analysis</h2>

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


train = pd.read_csv(r"C:\Users\yuvi\Downloads\train (1).csv")
test = pd.read_csv(r"C:\Users\yuvi\Downloads\test (1).csv")


# In[12]:


train


# In[ ]:





# In[6]:


train.isnull().sum()
print("Train Shape:",train.shape)
test.isnull().sum()
print("Test Shape:",test.shape)


# In[7]:


train.info()


# In[6]:


test.info()


# ### Data Dictionary
# 
# * Survived: 0 = No, 1 = Yes
# * pclass: Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd
# * sibsp: # of siblings / spouses aboard the Titanic
# * parch: # of parents / children aboard the Titanic
# * ticket: Ticket number
# * cabin: Cabin number
# * embarked: Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton
# 
# **Total rows and columns**
# 
# We can see that there are 891 rows and 12 columns in our training dataset.

# In[7]:


train.head(10)


# In[8]:


train.describe()


# In[9]:


test.describe()


# In[10]:


train.isnull().sum()


# In[9]:


test.isnull().sum()


# In[11]:


test.isnull().sum()
test["Survived"] = ""
test.head()


# # Data Visualization using Matplotlib and Seaborn packages.

# In[12]:


import matplotlib.pyplot as plt # Plot the graphes
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set() # setting seaborn default for plots


# # Bar Chart for Categorical Features 
# 
# * Pclass
# * Sex
# * SibSp ( # of siblings and spouse)
# * Parch ( # of parents and children)
# * Embarked
# * Cabin

# In[13]:


def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[14]:


bar_chart('Sex')
print("Survived :\n",train[train['Survived']==1]['Sex'].value_counts())
print("Dead:\n",train[train['Survived']==0]['Sex'].value_counts())


# The Chart confirms **Women more likely survivied than Men**.

# In[15]:


bar_chart('Pclass')
print("Survived :\n",train[train['Survived']==1]['Pclass'].value_counts())
print("Dead:\n",train[train['Survived']==0]['Pclass'].value_counts())


# The Chart confirms **1st class** more likely survivied than **other classes**.  
# The Chart confirms **3rd class** more likely dead than **other classes**

# In[16]:


bar_chart('SibSp')
print("Survived :\n",train[train['Survived']==1]['SibSp'].value_counts())
print("Dead:\n",train[train['Survived']==0]['SibSp'].value_counts())


# The Chart confirms a **person aboarded with more than 2 siblings or spouse** more likely survived.  
# The Chart confirms a **person aboarded without siblings or spouse** more likely dead

# In[17]:


bar_chart('Parch')
print("Survived :\n",train[train['Survived']==1]['Parch'].value_counts())
print("Dead:\n",train[train['Survived']==0]['Parch'].value_counts())


# The Chart confirms a **person aboarded with more than 2 parents or children more likely survived.**  
# The Chart confirms a **person aboarded alone more likely dead**

# In[18]:


bar_chart('Embarked')
print("Survived :\n",train[train['Survived']==1]['Embarked'].value_counts())
print("Dead:\n",train[train['Survived']==0]['Embarked'].value_counts())


# The Chart confirms a **person aboarded from C** slightly more likely survived.  
# The Chart confirms a **person aboarded from Q** more likely dead.  
# The Chart confirms a **person aboarded from S** more likely dead.  

# ## 4. Feature engineering

# Feature engineering is the process of using domain knowledge of the data
# to create features (**feature vectors**) that make machine learning algorithms work.  
# 
# feature vector is an n-dimensional vector of numerical features that represent some object.
# Many algorithms in machine learning require a numerical representation of objects,
# since such representations facilitate processing and statistical analysis.

# In[19]:


train.head()


# #### 4.1 how titanic sank?

# In[20]:


Image(url= "https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/t/5090b249e4b047ba54dfd258/1351660113175/TItanic-Survival-Infographic.jpg?format=1500w")


# In[21]:


train.head(10)


# In[22]:


train_test_data = [train,test] # combine dataset

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[23]:


train['Title'].value_counts()


# In[24]:


test['Title'].value_counts()


# #### Title Map
# 
# Mr : 0   
# Miss : 1  
# Mrs: 2  
# Others: 3  

# In[25]:


title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for dataset in train_test_data:
    dataset['Title'] = dataset["Title"].map(title_mapping)


# In[26]:


dataset.head()


# In[27]:


test.head()


# In[28]:


bar_chart('Title')


# In[29]:


# delete unnecessary feature from dataset
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)


# In[30]:


train.head()


# In[31]:


sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# In[32]:


bar_chart('Sex')


# In[33]:


test.head()


# In[34]:


train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace= True)
test["Age"].fillna(test.groupby('Title')['Age'].transform("median"), inplace= True)


# In[35]:


train.head(30)
#train.groupby("Title")["Age"].transform("median")


# In[36]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend() 
plt.show()

facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend() 
plt.xlim(10,50)


# Those who were **20 to 30 years old** were **more dead and more survived.**

# In[37]:


train.info()
test.info()


# **Binning**
# 
# Binning/Converting Numerical Age to Categorical Variable
# 
# feature vector map:
# * child: 0
# * young: 1
# * adult: 2
# * mid-age: 3
# * senior: 4

# In[38]:


train.head()


# In[39]:


for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4
# for dataset in train_test_data:
#     dataset.loc[]
#train[train['Age'].isin([23])]


# In[40]:


train.head()
bar_chart('Age')


# In[41]:


Pclass1 = train[train['Pclass'] == 1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass'] == 2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass'] == 3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1,Pclass2,Pclass3])
df.index = ['1st Class','2nd Class','3rd Class']
df.plot(kind = 'bar', stacked =  True, figsize=(10,5))
plt.show()
print("Pclass1:\n",Pclass1)
print("Pclass2:\n",Pclass2)
print("Pclass3:\n",Pclass3)


# more than 50 % of 1st class are from S embark.  
# more than 50 % of 2st class are from S embark.   
# more than 50 % of 3st class are from S embark.  
# 
# **fill out missing embark with S embark**

# In[42]:


for dataset in train_test_data:
    dataset['Embarked'] =  dataset['Embarked'].fillna('S')


# In[43]:


train.head()


# In[44]:


embarked_mapping = {'S':0,'C':1,'Q':2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


# In[45]:


# train["Fare"].fillna(train.groupby("Pclass")["Fare"])
# train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace = True)
# test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace = True)
# train.head(50)


# fill missing Fare with median fare for each Pclass
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
train.head(50)


# In[46]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4 )
facet.map(sns.kdeplot, 'Fare', shade = True)
facet.set(xlim = (0, train['Fare'].max()))
facet.add_legend()
plt.show()


# In[47]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
plt.xlim(0, 20)


# In[48]:


for dataset in train_test_data:
    dataset.loc[dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[dataset['Fare'] >= 100, 'Fare'] = 3


# In[49]:


train.head()


# In[50]:


train.Cabin.value_counts()


# In[51]:


for dataset in train_test_data:
    dataset['Cabin'] =  dataset['Cabin'].str[:1]


# In[52]:


Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[53]:


cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)


# In[54]:


# fill missing Fare with median fare for each Pclass
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)


# **family Size**

# In[55]:


train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1


# In[56]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'FamilySize',shade= True)
facet.set(xlim=(0, train['FamilySize'].max()))
facet.add_legend()
plt.xlim(0)


# In[57]:


family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)


# In[58]:


train.head()


# In[59]:


features_drop = ['Ticket','SibSp','Parch']
train = train.drop(features_drop, axis = 1)
test = test.drop(features_drop,axis=1)
train = train.drop(['PassengerId'], axis=1)


# In[60]:


train_data = train.drop('Survived', axis = 1)
target = train['Survived']
train_data.shape, target.shape


# In[61]:


train_data.head(10)


# # 5. Modelling

# In[62]:


# Importing Classifier Modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np


# In[63]:


train.info()


# # 6.Cross Validation(k-fold)

# In[64]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# In[65]:


clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[66]:


#learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
clf = [KNeighborsClassifier(n_neighbors = 13),DecisionTreeClassifier(),
       RandomForestClassifier(n_estimators=13),GaussianNB(),SVC(),ExtraTreeClassifier(),
      GradientBoostingClassifier(n_estimators=10, learning_rate=1,max_features=3, max_depth =3, random_state = 10),AdaBoostClassifier(),ExtraTreesClassifier()]
def model_fit():
    scoring = 'accuracy'
    for i in range(len(clf)):
        score = cross_val_score(clf[i], train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
        print("Score of Model",i,":",round(np.mean(score)*100,2))
#     round(np.mean(score)*100,2)
#     print("Score of :\n",score)
model_fit()


# In[67]:


clf1 = SVC()
clf1.fit(train_data, target)
test
test_data = test.drop(['Survived','PassengerId'], axis=1)
prediction = clf1.predict(test_data)
# test_data


# In[73]:


test_data['Survived'] = prediction
submission = pd.DataFrame(test['PassengerId'],test_data['Survived'])
submission.to_csv("Submission.csv")


# In[ ]:




