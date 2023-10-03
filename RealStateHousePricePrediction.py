#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib


# In[2]:


df1 = pd.read_csv("D:\\DataSets\\Bengaluru_House_Data.CSV")
df1.head()


# In[3]:


df1.shape #number of columns and rows in data set


# In[4]:


#examine are type feature
df1.groupby('area_type')['area_type'].agg('count')


# In[5]:


#just imagine avalability is not important to predict theprice then drop it
df2= df1.drop(['area_type','society','balcony','availability'],axis='columns')
df2.head()


# In[6]:


#data cleaning - handling the values
#check is there null values
df2.isnull()


# In[7]:


# then get the sum of the null values
df2.isnull().sum()


# In[8]:


#drop null value columns
df3 = df2.dropna()
df3.isnull().sum()


# In[9]:


df3.shape


# In[10]:


#explore size variable
df3['size'].unique()


# In[11]:


#create new colomn with tokenize like 2 bhk - 2 
df3['bhk']= df3['size'].apply(lambda x : int(x.split(' ')[0]))
df3.head()


# In[12]:


df3['bhk'].unique()


# In[13]:


#explore total squre
df3['total_sqft'].unique()
#in there we can see number range


# In[14]:


#convert no range into single number 1. get average
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[15]:


df3[~df3['total_sqft'].apply(is_float)] # ~ used to look not valid float


# In[16]:


#function to take range and convert into average
def convert_sqrt_to_num (x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
    


# In[17]:


convert_sqrt_to_num('2100 - 2850')


# In[18]:


#apply function to column
df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqrt_to_num)
df4.head()


# In[19]:


#access 30 row
df4.loc[30]


# In[20]:


df5 = df4.copy()
# add another coulmn price per squre feet
df5['price_per_sqrt'] = df5['price']*100000/df5['total_sqft']
df5.head()


# In[21]:


#explore location 
#how many location
df5.location.unique()
len(df5.location.unique())


# In[22]:


#figure out how many data points are available for location
#1st strip any extra spaces in location
df5.location = df5.location.apply(lambda x: x.strip())

location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats


# In[23]:


len(location_stats[location_stats<=10])


# In[24]:


location_stat_less_than_10=location_stats[location_stats<=10]
location_stat_less_than_10


# In[25]:


len(df5.location.unique())


# In[26]:


#location_stat_less_than_10 print as other
df5.location = df5.location.apply(lambda x: 'other' if x in location_stat_less_than_10 else x)
len(df5.location.unique())


# In[27]:


df5.head(10)


# In[28]:


#remove outliers - unormal data
df5[df5.total_sqft/df5.bhk<300].head()


# In[29]:


df5.shape


# In[30]:


df6 = df5[~(df5.total_sqft/df5.bhk<300)]
df6.shape


# In[31]:


df6.price_per_sqrt.describe()


# In[32]:


#outlier removel
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqrt)
        st = np.std(subdf.price_per_sqrt)
        reduced_df = subdf[(subdf.price_per_sqrt>(m-st)) & (subdf.price_per_sqrt<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
df7.shape


# In[33]:


#check 2bd room price is less or more then 3bd room price like wise then draw scatter plot
def plot_scatter_chart (df, location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price_per_sqrt,color = 'blue', label = '2 Bhk', s= 50)
    plt.scatter(bhk3.total_sqft,bhk3.price_per_sqrt,marker = '+', color = 'green', label = '3 Bhk', s= 50)
    plt.xlabel("Total Squre Feet Area")
    plt.ylabel("Price")
    plt.title(location)
    plt.legend()
plot_scatter_chart(df7,"Hebbal")


# In[34]:


#removing outlers
#now we remove those 2bhk aparetmnets whose price_per_sqft is less than mean price_per_sqft of 1bhk apartement

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats [bhk] = {
                'mean': np.mean(bhk_df.price_per_sqrt),
                'std': np.std(bhk_df.price_per_sqrt),
                'count':bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices,bhk_df[bhk_df.price_per_sqrt<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

df8 = remove_bhk_outliers(df7)
df8.shape


# In[35]:


plot_scatter_chart(df8,"Hebbal")


# In[36]:


import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqrt,rwidth=0.8)
plt.xlabel("Price Per Squre Feet")
plt.ylabel("Count")


# In[37]:


#bathroom feature
df8.bath.unique()


# In[38]:


df8[df8.bath>10]


# In[39]:


plt.hist(df8.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")


# In[40]:


df8[df8.bath>df8.bhk+2] # having more than 2 bhk


# In[41]:


df9 = df8[df8.bath<df8.bhk+2]
df9.shape


# In[42]:


df10 = df9.drop(['size','price_per_sqrt'],axis='columns')
df10.head(3)


# In[43]:


#machine lerning model can't interprit text data then we need to convert location column as numerical i call dummies
dummies=pd.get_dummies(df10.location)
dummies.head(3)


# In[44]:


#aapend them into main data frame
df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df11.head(3)


# In[45]:


#now drop the location column
df12 = df11.drop('location',axis='columns')
df12.head(2)


# In[46]:


df12.shape


# In[47]:


#x variable should contain only independant variable
#dependant variable is price then it should want to drop
X = df12.drop('price', axis='columns')
X.head()


# In[48]:


y = df12.price
y.head()


# In[49]:


#dive data set to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)


# In[50]:


from sklearn.linear_model import LinearRegression


# In[51]:


lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)


# In[52]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(),X, y, cv=cv)


# In[53]:


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridserchcv(X,y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'normalize': [True,False]
            }
        },
        'lasso':{
            'model': Lasso(),
            'params':{
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree':{
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5,test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =GridSearchCV(config['model'],config['params'], cv= cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridserchcv(X,y)
    


# In[54]:


import pandas as pd
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridserchcv(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                # Remove 'normalize' parameter
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['mse', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

# Assuming you have defined X and y somewhere before calling this function
find_best_model_using_gridserchcv(X, y)


# In[55]:


X.columns


# In[57]:


#price prediction

def predict_price (location,sqft,bath,bhk):
    loc_index = np.where(X.columns == location) [0][0]
    
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
        
    return lr_clf.predict([x])[0]


# In[58]:


predict_price('1st Phase JP Nagar',1000, 2,2)


# In[60]:


predict_price('Indira Nagar',1000,2,2)


# In[61]:


#export model
import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f) # pass your model your classifier as an argument


# In[62]:


#export columns infomation to jason file
import json
columns = {
    'data_columns': [col.lower() for col in X.columns]
}
with open("columns.json", "w") as f:
    f.write(json.dumps(columns))


# In[ ]:




