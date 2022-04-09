#!/usr/bin/env python
# coding: utf-8

# 

# # NAME : BILLA PENCHALA SAI DINESH 

# # SPARK FOUNDATION INTERSHIP

# # TASK 1 - PREDICT THE PERCENTAGE OF STUDENTS BASED ON NO: OF STUDY HOURS

# In[ ]:





# # IMPORTING THE DATASHEET WITH LIBRARIES

# In[2]:


#We are using the data to import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#we are reading the data from the website
df=pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")


# In[4]:


df


# In[6]:


df.head()


# In[7]:


df.tail()


# In[9]:


df.shape


# In[10]:


df.info()


# In[12]:


df.describe()


# # step-2 Data visualization

# In[13]:


#ploting thr data in graph
plt.rcParams["figure.figsize"] = [15,9]
df.plot(x="Hours" , y="Scores" , style="*" , color="blue" , markersize=10)
plt.title("Hours VS Percentage")
plt.xlabel("hours studied")
plt.ylabel("percentage scored")
plt.grid()
plt.show()


# # from the above graph, we can observe that there is a linear relation between the "hours studied" and "percentage score". So we can use the linear regression supervised machine model on it to predict further values.

# In[14]:


#we can also use .corr to determine the correlation betweeen the variables.
df.corr()


# # STEP 3 - PREPARATION OF DATA

# In[15]:


#here we are divind the data set into two parts -  TEST DATA AND TRAINING DATA


# In[16]:


#Using the iloc function we will deivide the data
x = df.iloc[:, :1].values
y = df.iloc[:, 1:].values


# In[17]:


x


# In[18]:


y


# In[19]:


#splitting the data into training data and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                            test_size=0.25, random_state=0)


# # STEP 4 - TRAINING THE ALGORITHM

# In[20]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)


# # STEP 5 - VISUALIZING THE MODE

# In[21]:


line = model.coef_*x +model.intercept_

#plotting for the training data
plt.rcParams["figure.figsize"] = [15,9]
plt.scatter(x_train, y_train, color="blue")
plt.plot(x, line, color="red");
plt.xlabel("hours studied")
plt.ylabel("percentage scored")
plt.grid()
plt.show()


# In[22]:


#plotting for the testing data
plt.rcParams["figure.figsize"] = [15,9]
plt.scatter(x_test, y_test, color="blue")
plt.plot(x, line, color="red");
plt.xlabel("hours studied")
plt.ylabel("percentage scored")
plt.grid()
plt.show()


# # STEP 6 - MAKING THE PREDICTIONS

# In[23]:


print(x_test) #testing data - In Hours
y_pred = model.predict(x_test) #predicting the scores


# In[24]:


#comparing thr actual vs predicted
y_test


# In[25]:


y_pred


# In[26]:


#comparing actual vs predicted
comp = pd.DataFrame({ 'Actual':[y_test],'predicted':[y_pred] })
comp


# In[27]:


#testing with your own data

hours = 9.25
own_pred = model.predict([[hours]])
print("the predicted score if a person studies for", hours, "hours is", own_pred[0])


# # hence here it can be predicted that if a person studies for 9.25hrs then the score is 93.89272889

# # STEP 7 - EVALUVATING THE MODEL

# In[29]:


from sklearn import metrics

print("mean absolute Error:", metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:




