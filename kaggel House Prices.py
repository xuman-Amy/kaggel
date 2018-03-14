
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  #统计绘图 

from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy import stats  #统计

import warnings
warnings.filterwarnings('ignore')
#画图直接显示
get_ipython().magic('matplotlib inline')


# In[4]:


#bring in the six packs
df_train = pd.read_csv('G:Machine learning\\kaggel\\house prices\\train.csv')
df_test = pd.read_csv('G:Machine learning\\kaggel\\house prices\\test.csv')


# In[5]:


#check the decoration
#数据.columns 各列名称 分析有哪些数据，可以将数据分为numerical (数值型)和categorical(类别型)
df_train.columns


# In[6]:


#describe函数用来数据的快速统计汇总
df_train['SalePrice'].describe()


# In[7]:


#seaborn 用法  https://zhuanlan.zhihu.com/p/24464836
#seaborn的displot()集合了matplotlib的hist()与核函数估计kdeplot的功能，
#增加了rugplot分布观测条显示与利用scipy库fit拟合参数分布的新颖用途

sns.distplot(df_train['SalePrice'])


# In[8]:


#show skewness and Kurtosis  偏态和峰度
print("Skewness : %f " % df_train['SalePrice'].skew())
print("Kurtosis : %f " % df_train['SalePrice'].kurt())
                                  


# In[9]:


#scatter plot  Grlivearea / SalePrice

var = 'GrLivArea'
#pd.concat 函数 可以将数据根据不同的轴作简单的融合 axis = 0-->代表行  axis = 1 --> 代表列

data = pd.concat([df_train['SalePrice'],df_train[var]],axis = 1)
data.plot.scatter(x = var, y = 'SalePrice',ylim = (0,800000));


# In[10]:


#scatter plot saleprice / totalbsmtsf
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'],df_train[var]],axis = 1)
data.plot.scatter(x = var, y = 'SalePrice',ylim = (0,800000))


# In[11]:


#box plot overallqual / saleprice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'],df_train['OverallQual']],axis = 1)
f,ax = plt.subplots(figsize = (8,6)) #subplots 创建一个画像(figure)和一组子图(subplots)。 
fig = sns.boxplot(x = var,y = 'SalePrice',data = data)
fig.axis (ymin = 0,ymax = 800000)


# In[12]:


#boxplot saleprice / yearbuilt
var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'],df_train['YearBuilt']],axis = 1)
f, ax = plt.subplots(figsize = (25,10))
fig = sns.boxplot(x = var,y = 'SalePrice',data = data)
fig.axis(ymin = 0,yamx = 800000)
plt.xticks(rotation = 90) #x轴标签 转90度


# In[13]:


#correlation matrix  相关矩阵
corrmat = df_train.corr()
f ,ax = plt.subplots(figsize = (12,9))
sns.heatmap(corrmat,vmax = 8,square = True,cmap = 'hot')


# In[14]:


#saleprice correlation matrix
k = 10
cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index  #取出与saleprice相关性最大的十项
cm = np.corrcoef(df_train[cols].values.T)  #相关系数 
sns.set(font_scale = 1.25)
hm = sns.heatmap(cm,cbar = True,annot = True,square = True ,fmt = '.2f',annot_kws = {'size': 10},yticklabels = cols.values,xticklabels = cols.values)
plt.show()


# In[15]:


#scatterplot 
sns.set()
cols = ['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']
sns.pairplot(df_train[cols],size = 2.5)



# In[16]:


#missing data
#pandas.isnull() 判断数据是否为空 返回false / true
#sort_values()
total = df_train.isnull().sum().sort_values(ascending = False)
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat([total,percent],axis = 1,keys = ['Total','Percent'])
missing_data.head(20)


# In[17]:


df_train = df_train.drop(missing_data[(missing_data['Total'] > 1)].index,axis = 1)


# In[18]:


df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)


# In[19]:


df_train.isnull().sum().max()


# In[20]:


#standardizing data --> converting data  values to have mean of 0 and standard deviation of 1 
#http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# fit : compute mean and std deviation 
#transform : Perform standardization by centering and scaling
#np.newaxis 增加新维度
#argsort() Returns the indices that would sort an array.将x中的元素从小到大排列，提取其对应的index(索引)，然后输出
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range(low) of the distribution :','\n',low_range)
print ('outer range (high) of the distribution :','\n',high_range)
sns.distplot(saleprice_scaled)


# In[21]:


var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'],df_train[var]],axis = 1)
data.plot.scatter(x = var, y = 'SalePrice',ylim =(0,800000))


# In[ ]:


#delete point
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299 ].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)


# # 5 Getting hard core
# 
# According to Hair et al. (2013), four assumptions should be tested:
# 
# Normality (正态性)- When we talk about normality what we mean is that the data should look like a normal distribution. This is important because several statistic tests rely on this (e.g. t-statistics). In this exercise we'll just check univariate normality for 'SalePrice' (which is a limited approach). Remember that univariate normality doesn't ensure multivariate normality (which is what we would like to have), but it helps. Another detail to take into account is that in big samples (>200 observations) normality is not such an issue. However, if we solve normality, we avoid a lot of other problems (e.g. heteroscedacity) so that's the main reason why we are doing this analysis.
# 
# Homoscedasticity（方差齐性） - I just hope I wrote it right. Homoscedasticity refers to the 'assumption that dependent variable(s) exhibit equal levels of variance across the range of predictor variable(s)' (Hair et al., 2013). Homoscedasticity is desirable because we want the error term to be the same across all values of the independent variables.
# 
# Linearity（线性）- The most common way to assess linearity is to examine scatter plots and search for linear patterns. If patterns are not linear, it would be worthwhile to explore data transformations. However, we'll not get into this because most of the scatter plots we've seen appear to have linear relationships.
# 
# Absence of correlated errors （无相关错误） - Correlated errors, like the definition suggests, happen when one error is correlated to another. For instance, if one positive error makes a negative error systematically, it means that there's a relationship between these variables. This occurs often in time series, where some patterns are time related. We'll also not get into this. However, if you detect something, try to add a variable that can explain the effect you're getting. That's the most common solution for correlated errors.

# In[22]:


# in the search for normality
#histogram and normal probability plot  直方图和正态概率图
sns.distplot(df_train['SalePrice'],fit = norm) #fit 控制拟合的参数分布图形
fig = plt.figure()
# probplot :Calculate quantiles for a probability plot, and optionally show the plot. 计算概率图的分位数
res = stats.probplot(df_train['SalePrice'],plot = plt)


# In[23]:


# in case of positive skewness, log transformations usually works well. 
#直方图看出不是正态分布，有poistive skewness(正偏态 高峰偏左)---> log transformation 
df_train['SalePrice'] = np.log(df_train['SalePrice'])
sns.distplot(df_train['SalePrice'],fit = norm)
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'],plot = plt)


# In[24]:


# check GrLivArea
sns.distplot(df_train['GrLivArea'],fit = norm)
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'],plot = plt)


# In[25]:


df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
sns.distplot(df_train['GrLivArea'],fit = norm)
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'],plot = plt)


# In[26]:


#TotalBsmtSF 
sns.distplot(df_train['TotalBsmtSF'],fit = norm)
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'],plot = plt)


# Ok, now we are dealing with the big boss. What do we have here?
# 
# Something that, in general, presents skewness.
# A significant number of observations with value zero (houses without basement).
# A big problem because the value zero doesn't allow us to do log transformations.
# 
# To apply a log transformation here, we'll create a variable that can get the effect of having or not having basement (binary variable). Then, we'll do a log transformation to all the non-zero observations, ignoring those with value zero. This way we can transform data, without losing the effect of having or not basement.

# In[27]:


#create column for new varible 
#if area > 0 ,it gets 1; for area == 0 it gets 0
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']),index = df_train.index)
df_train['HasBsmt']  = 0
df_train.loc[df_train['TotalBsmtSF'] > 0 ,'HasBsmt'] = 1
df_train.loc[df_train['HasBsmt'] == 0 ,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])


# In[28]:


sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'],fit = norm)
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF'] > 0 ]['TotalBsmtSF'],plot = plt)


# In[29]:


#scatter plot
plt.scatter(df_train['SalePrice'],df_train['GrLivArea'])


# In[30]:


plt.scatter(df_train[df_train['TotalBsmtSF']>0] ['TotalBsmtSF'],df_train[df_train['TotalBsmtSF'] > 0] 
['SalePrice'])


# In[32]:


#convert categorical variable into dummy
df_train = pd.get_dummies(df_train)
df_train


# In[ ]:





# In[ ]:





# In[ ]:




