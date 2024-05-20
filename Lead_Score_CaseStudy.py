#!/usr/bin/env python
# coding: utf-8

# ## Lead Scoring Case Study

# ## Problem statement
# An education company named X Education sells online courses to industry professionals. On any given day, many professionals who are interested in the courses land on their website and browse for courses. The company markets its courses on several websites and search engines like Google. Once these people land on the website, they might browse the courses or fill up a form for the course or watch some videos. When these people fill up a form providing their email address or phone number, they are classified to be a lead. The typical lead conversion rate at X education is around 30%. To make this process more efficient, the company wishes to identify the most potential leads, also known as ‘Hot Leads’. If they successfully identify this set of leads, the lead conversion rate should go up as the sales team will now be focusing more on communicating with the potential leads rather than making calls to everyone.
# 
# ### Goal
# Build a logistic regression model to assign a lead score between 0 and 100 to each of the leads which can be used by the company to target potential leads. A higher score would mean that the lead is hot, i.e. is most likely to convert whereas a lower score would mean that the lead is cold and will mostly not get converted. There are some more problems presented by the company which your model should be able to adjust to if the company's requirement changes in the future so you will need to handle these as well. 

# ## Step 1: Importing  Data and Inspecting the Dataframe

# In[1]:


# Importing the IMP library
import pandas as pd
import numpy as np


#Imprting the library to avoid wanings
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#Importing the Visualization Library
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#Reading The given datasets
ld_df=pd.read_csv("Leads.csv")


# In[4]:


# Let's see the head of dataset
ld_df.head()


# In[5]:


ld_df.describe() # describe the stastical information of all numeric columns


# In[6]:


# Let's see the type of each column
ld_df.info()


# In[7]:


# Select all non-numeric columns
s_df = ld_df.select_dtypes(include='object')

# Find out columns that have "Select"
s = lambda x: x.str.contains('Select', na=False)
l = s_df.columns[s_df.apply(s).any()].tolist()
print (l)


# In[8]:


# select all the columns that have a "Select" entry
sel_cols =['Specialization', 'How did you hear about X Education', 'Lead Profile', 'City']
# replace values
ld_df[sel_cols] = ld_df[sel_cols].replace('Select', np.NaN)


# In[9]:


# Check NULL values
ld_df_null=round(100*ld_df.isnull().mean(),2)
ld_df_null


# 
# Asymmetrique Activity Index    has                  45.65%
# Asymmetrique Profile Index    has                   45.65%
# Asymmetrique Activity Score    has                  45.65%
# Asymmetrique Profile Score     has                  45.65%
# Lead Quality                  has                   51.59%
# 
# Lead Profile      has                               74.19%
# How did you hear about X Education has 78% null values
# it is better to drop these columns
# 

# In[10]:


# Check NULL values and drop mean > 0.40
ld_df= ld_df.drop(ld_df.columns[ld_df.isnull().mean() >= 0.40], axis = 1)


# In[11]:


#Prospect ID is A unique ID with which the customer is identified so drop it
ld_df=ld_df.drop("Prospect ID",axis=1)


# In[12]:


#"Lead Number" is A unique number so drop it
ld_df=ld_df.drop("Lead Number",axis=1)


# In[13]:


# Let's check the dimensions of the dataframe
ld_df.shape


# In[14]:


# Check all columns
ld_df.columns


# In[15]:


# Make a list of continuous values by appending
# Make a list of catogerical values by appending 
cont_cols=[]
cat_cols=[]
for i in ld_df.columns:
    if ld_df[i].nunique()>30:
        print(i,ld_df[i].nunique(),  "----cont_cols")
        cont_cols.append(i)
    else:
        print(i,ld_df[i].nunique(),  "----cat_cols")
        cat_cols.append(i)


# In[16]:


# Print catogrical values
# Print continuous values
print(cat_cols)
print(cont_cols)


# In[17]:


# Continous values list
cont_cols=['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit', ]


# In[18]:


# Catogrical values list
cat_cols=['Lead Origin', 'Lead Source', 'Do Not Email', 'Do Not Call', 'Converted', 'Last Activity', 'Specialization', 
         'What is your current occupation', 'What matters most to you in choosing a course', 'Search', 'Newspaper Article',
         'X Education Forums', 'Newspaper', 'Digital Advertisement', 'Through Recommendations', 'Tags', 'City', 
         'A free copy of Mastering The Interview', 'Last Notable Activity','Country']


# In[19]:


# Catogrical values list along with value counts
for i in cat_cols:
    print(i)
    print()
    print(ld_df[i].value_counts())
    print()
    print("percentage",100*ld_df[i].value_counts(normalize=True))
    print("--------------------------------------------------")


#  Replacing the null values in categorical columns by mode of that respective column

# In[20]:


for i in cat_cols:
    if ld_df[i].isnull().sum()>0:
        value=ld_df[i].mode()[0]
        ld_df[i]=ld_df[i].fillna(value)


#  Replacing the null values in continuous columns by median of that respective column

# In[21]:


for i in cont_cols:
    if ld_df[i].isnull().sum()>0:   
        value=ld_df[i].median()
        ld_df[i]=ld_df[i].fillna(value)


# In[22]:


#now check all the null values are replaced or not
ld_df.isnull().sum()


# In[23]:


ld_df.columns


# In[24]:


ld_df.describe()


# In[25]:


ld_df.info()


# ## Step 2. Exploratory Data Analytics
# ## Univariate Analysis¶
# ## Converted

# Now plot the bar graph for cat_cols columns to visualise the value count of it

# In[26]:


plt.figure(figsize=(12,5))
for i in cat_cols:
    print(i)
    ld_df[i].value_counts().plot.barh(width=.5)
    plt.show()


# From the above visualisation following columns have only one value "No" in all the rows, we can drop theses columns :
# 
#  a) Magazine
#  b) Receive More Updates About Our Courses
#  c) Update me on Supply Chain Content
#  d) Get updates on DM Content
#  e) I agree to pay the amount through cheque

# In[27]:


ld_df = ld_df.drop(['Magazine', 'Receive More Updates About Our Courses', 'Update me on Supply Chain Content', 
                          'Get updates on DM Content', 'I agree to pay the amount through cheque'], axis=1)


# Based on the above visualization, we can drop the variables which are not significant for analysis and will not give any information to the model.

# In[28]:


ld_df = ld_df.drop(['Country','What matters most to you in choosing a course','Search','Newspaper Article',
                          'X Education Forums','Newspaper','Digital Advertisement'],1)


# the columns Through Reccomedations,  Dont not call are skwed or have unbalance data which will craete biasness so drop these features

# In[29]:


ld_df = ld_df.drop(['Through Recommendations','Do Not Call'],1)


# In[30]:


ld_df.shape


# In[31]:


ld_df.columns


# let's check the value count of remaining columns

# In[32]:


# Creating Catogrical lists1,2,3,4 for further analysis 
cat_col1=['Lead Origin','Lead Source']
cat_col2=['Last Notable Activity', 'Last Activity']
cat_col3=['Do Not Email','A free copy of Mastering The Interview','What is your current occupation' ]
cat_col4=['Specialization','Tags', 'City']


# In[33]:


# Creating Continuous list for further analysis 
cont_cols=[ 'TotalVisits',
       'Total Time Spent on Website', 'Page Views Per Visit']


# In[34]:


# Creating count plot for Catogrical values list-1
plt.figure(figsize=(25,10))
for i in enumerate(cat_col1):
    plt.subplot(2,3,i[0]+1)
    sns.countplot(i[1], hue = 'Converted', data = ld_df)
    plt.xlabel(i,size = 14)
    plt.xticks(rotation = 90)


# To improve the overall lead conversion rate, we need to focus on increasing the conversion rate of 'API' and 'Landing Page Submission' and also increasing the number of leads from 'Lead Add Form'

# In[35]:


# Replace values for consistency like google to Google, Social Media and Others in Lead Source column for proper analysis
ld_df['Lead Source'] = ld_df['Lead Source'].replace('google','Google')
ld_df['Lead Source'] = ld_df['Lead Source'].replace('Facebook','Social Media')
ld_df['Lead Source'] = ld_df['Lead Source'].replace(['bing','Click2call','Press_Release',
                                                     'youtubechannel','welearnblog_Home',
                                                     'WeLearn','blog','Pay per Click Ads',
                                                    'testone','NC_EDM'] ,'Others')


# In[36]:


plt.figure(figsize=(10,5))

#Visualising using Countplot

count_fig=sns.countplot(ld_df['Lead Source'], hue=ld_df['Converted'])
count_fig.set_xticklabels(count_fig.get_xticklabels(),rotation=45)

#Formatting the plot
plt.title("Leads Conversion based on Tags",fontsize=12)
plt.show()


# In[37]:


# Creating count plot for Catogrical values list-2
plt.figure(figsize=(25,10))
for i in enumerate(cat_col2):
    plt.subplot(2,3,i[0]+1)
    sns.countplot(i[1], hue = 'Converted', data = ld_df)
    plt.xlabel(i,size = 14)
    plt.xticks(rotation = 90)


# categories that has less occurance on the Last Notable Activity may be replaced as other notable activity and 
# these column is similar to last activity so drop this col
# The conversion rateis maximum of lst activity as "Email Opened" 
# The conversion rate of SMS sent as last activity is maximum
# so we have to make a call to the lead who has opened their email and to whom sms sent to increase the conversion rate
# 

# In[38]:


# Replaced specific values to Other_Notable_activity in column Last Notable Activity
ld_df['Last Notable Activity'] = ld_df['Last Notable Activity'].replace(['Had a Phone Conversation','Email Marked Spam',
                                                                       'Unreachable','Unsubscribed','Email Bounced',
                                                                       'Resubscribed to emails','View in browser link Clicked',
                                                                       'Approached upfront', 'Form Submitted on Website',
                                                                       'Email Received'],'Other_Notable_activity')
ld_df = ld_df.drop(['Last Activity'],1)


# In[39]:


# Creating count plot for Catogrical values list-3
plt.figure(figsize=(25,10))
for i in enumerate(cat_col3):
    plt.subplot(2,3,i[0]+1)
    sns.countplot(i[1], hue = 'Converted', data = ld_df)
    plt.xlabel(i,size = 14)
    plt.xticks(rotation = 90)


# most of  leads search on google, unemloyed, and sent sms, and opened their Email are converted,

# In[40]:


# Creating count plot for Catogrical values list-4
plt.figure(figsize = (25, 10))
for i in enumerate(cat_col4):
    plt.subplot(2,4,i[0]+1)
    print(i)
    sns.countplot(i[1], hue = 'Converted', data = ld_df)
    plt.xticks(rotation = 90)


# In specialization management course is divided into subgroups so let it bring in one as management

# In[41]:


# Replaced specific values to Management_Specializations in column Specialization
ld_df['Specialization'] = ld_df['Specialization'].replace(['Finance Management','Human Resource Management',
                                                           'Marketing Management','Operations Management',
                                                           'IT Projects Management','Supply Chain Management',
                                                    'Healthcare Management','Hospitality Management',
                                                           'Retail Management'] ,'Management_Specializations') 


# In[42]:


plt.figure(figsize=(10,5))

#Visualising using Countplot

count_fig=sns.countplot(ld_df['Specialization'], hue=ld_df['Converted'])
count_fig.set_xticklabels(count_fig.get_xticklabels(),rotation=45)

#Formatting the plot
plt.title("Leads Conversion based on Tags",fontsize=12)
plt.show()


# In Tag column different tags are given from that we can make one tag as a not eligible or others 

# In[43]:


# Replaced specific values to Others_or_not_eligible in column Tags
ld_df['Tags'] = ld_df['Tags'].replace(['In confusion whether part time or DLP','Diploma holder (Not Eligible)',
                                     'Approached upfront','Graduation in progress','number not provided', 'opp hangup','Lateral student',
                                    'Recognition issue (DEC approval)',
                                    'University not recognized','switched off','Already a student','Not doing further education',
                                     'invalid number','wrong number given',
                                 'Interested  in full time MBA''Closed by Horizzon', 'Lost to EINS',
                                 'Lost to Others', 'in touch with EINS '], 'Others_or_not_eligible')


# some of the tags are could be converted to positive leads put it in one category

# In[44]:


# Replaced specific values to Could_be_Potential in column Tags
ld_df['Tags'] = ld_df['Tags'].replace(['Want to take admission but has financial problems', 'Still Thinking',
                                    'Shall take in the next coming month','Interested in Next batch','Ringing','Busy'],'Could_be_Potential' )


# In[45]:


plt.figure(figsize=(10,5))

#Visualising using Countplot

count_fig=sns.countplot(ld_df['Tags'], hue=ld_df['Converted'])
count_fig.set_xticklabels(count_fig.get_xticklabels(),rotation=45)

#Formatting the plot
plt.title("Leads Conversion based on Tags",fontsize=12)
plt.show()


# ## Handling Outliers:

# Next plot the boxplot for numerical/continuous columns to know the outliers

# In[46]:


for i in cont_cols:
    print(i)
    print("Max value is",ld_df[i].describe()["max"])
    plt.figure(figsize=(10,5))
    sns.boxplot(ld_df[i])
    plt.show()


# After plotting the boxplot for continuous columns it shows the point(values) beyond the upper limit(third quartile) these are the outliers
# 
# outiers are found in the following columns
# TotalVisits Max value is 251.0
# Page Views Per Visit -- Max value is 55.0

# In[47]:


Outlier_cols=["TotalVisits","Page Views Per Visit"]


# In[48]:


for i in Outlier_cols:                          ##Handling outliers using caping method
    Q1=ld_df[i].describe()["25%"]
    Q3=ld_df[i].describe()["75%"]
    IQR=Q3-Q1
    Upper_limit=Q3+1.5*IQR
    lower_limit=Q1-1.5*IQR
    
    ld_df[i]=np.where(ld_df[i]<lower_limit,lower_limit,ld_df[i])
    ld_df[i]=np.where(ld_df[i]>Upper_limit,Upper_limit,ld_df[i])


# In[49]:


for i in cont_cols:
    print(i)
    print("Max value is",ld_df[i].describe()["max"])
    plt.figure(figsize=(10,5))
    sns.boxplot(ld_df[i])
    plt.show()


# Outliers are handled by using caping and flooring method

# In[50]:


for i in cont_cols:
    print(i)
    plt.figure(figsize=(10,5))
    sns.boxplot(y=ld_df[i], x='Converted',data=ld_df,hue='Converted')
    plt.show()


# Median for converted and not converted leads are the same for  'Page Views Per Visit' and for total visits
# Leads spending more time on the website are more likely to be converted.

# ## Step 3: Data Preparation
# ### Converting some binary variables (Yes/No) to 0/1¶

# In[51]:


ld_df.columns


# In[52]:


columns=['Lead Origin', 'Lead Source', 'Do Not Email', 'Converted',
       'TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit',
       'Specialization', 'What is your current occupation', 'Tags', 'City',
       'A free copy of Mastering The Interview', 'Last Notable Activity']


# In[53]:


# To convert binary variable (Yes/No) to 0/1
ld_df['Do Not Email'] = ld_df['Do Not Email'].map({'Yes': 1, 'No': 0})
ld_df['A free copy of Mastering The Interview'] = ld_df['A free copy of Mastering The Interview'].map({'Yes': 1, 'No': 0})


# In[54]:


ld_df.head()


# In[55]:


ld_df.info()


# For categorical variables with multiple levels, create dummy features (one-hot encoded)

# In[56]:


# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy_1 = pd.get_dummies(ld_df[['Lead Origin', 'Lead Source', 'Last Notable Activity']], drop_first=True)

# Adding the results to the master dataframe
ld_df = pd.concat([ld_df, dummy_1], axis=1)


# In[57]:


ld_df=ld_df.drop(['Lead Origin', 'Lead Source', 'Last Notable Activity'],1)


# In[58]:


# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy_2 = pd.get_dummies(ld_df[['Specialization','What is your current occupation','Tags','City']], drop_first=True)
# Adding the results to the master dataframe
ld_df = pd.concat([ld_df, dummy_2], axis=1)


# In[59]:


ld_df=ld_df.drop(['Specialization','What is your current occupation','Tags','City'],1)


# In[60]:


ld_df.head()


# In[61]:


ld_df.info()


# Now you can see that you have all variables as numeric.

# In[62]:


# Checking the Rate of Conversion
Converted = (sum(ld_df['Converted'])/len(ld_df['Converted'].index))*100
Converted


# ## Step 4: Test-Train Split

# In[63]:


# Assigning value to target variable
X=ld_df.drop(["Converted"],axis=1)
y=ld_df["Converted"]


# In[64]:


# 70-30 ratio of split between train-test with random_state
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,test_size=0.3,random_state=42)


# In[65]:


X_train.head()


# ## Step 5: Feature Scaling

# In[66]:


from sklearn.preprocessing import StandardScaler


# In[67]:


# Using Standard Scaler() and make X_train learn using fit_transform
scaler = StandardScaler()

X_train[['Total Time Spent on Website','Page Views Per Visit','TotalVisits']] = scaler.fit_transform(X_train[['Total Time Spent on Website',
                                                                                            'TotalVisits','Page Views Per Visit']])

X_train.head()


# Conversion rate is about 38%

# In[68]:


# Correlation between different numerical variables for both the Converted and not-converted cases
correlation = ld_df.corr()
correlation


# ### Step 6: Model Building
# 

# Running First Training Model

# In[69]:


import statsmodels.api as sm


# In[70]:


# logistic regression model_1
X_train=sm.add_constant(X_train)
lm_1=sm.GLM(y_train,X_train, family=sm.families.Binomial())


# In[71]:


lm_1.fit().summary()


# ### Step 7: Feature Selection Using RFE

# In[72]:


from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()


# In[73]:


from sklearn.feature_selection import RFE
rfe=RFE(lg,n_features_to_select=15) # running Recursive Feature Elimination(RFE) with 15 variables as output
rfe.fit(X_train,y_train)


# In[74]:


# Check which are supported
rfe.support_


# In[75]:


# List the Zip columns,support and ranking for it
list(zip(X_train.columns,rfe.support_, rfe.ranking_))


# In[76]:


# Put all the columns selected by RFE in the variable 'col'

col=X_train.columns[rfe.support_]
len(col)


# In[77]:


X_train.columns[~rfe.support_]


# ##### Assessing the model with StatsModels

# In[78]:


# Assign constant to X_train_sm, create Generalized Linear Model
X_train_sm=sm.add_constant(X_train[col])
lm_2=sm.GLM(y_train,X_train_sm, family=sm.families.Binomial())
res=lm_2.fit()
print(res.summary())


# In[79]:


# Note that column "Lead Source_Welingak Website" has very high P value.


# In[80]:


# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[81]:


y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]


# Creating a dataframe with the actual lead converted and  the predicted conversion probabilities

# In[82]:


# Map columns converted_lead and conv_lead_prob to corresponding values
y_train_pred_final=pd.DataFrame({'converted_lead':y_train.values, 'conv_lead_prob':y_train_pred})
y_train_pred_final['lead_No']=y_train.index
y_train_pred_final.head()


# Creating new column 'predicted' with 1 if conv_lead_prob > 0.5 else 0

# In[83]:


# predicted column will be 1 if value of y_train_pred_final 'conv_lead_prob' is greater than 0.5
y_train_pred_final["predicted"]=y_train_pred_final['conv_lead_prob'].map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[84]:


from sklearn import metrics


# In[85]:


# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.converted_lead, y_train_pred_final.predicted )
print(confusion)


# In[86]:


#Lets check Accuracy
print("accuracy_score")
print(metrics.accuracy_score(y_train_pred_final["converted_lead"],y_train_pred_final["predicted"]))


# In[87]:


# checking correlation
ld_df.corr()


# Checking VIF

# In[88]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[89]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif=pd.DataFrame()
vif["Features"]=X_train[col].columns
z=X_train[col].shape[1]
vif["VIF"]=[variance_inflation_factor(X_train[col].values,i) for i in range(z)]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# 

# In[90]:


#Lead Source_Welingak Website has high p value as mentioned in note above so let's  drop it.
col=col.drop(["Lead Source_Welingak Website"],1)


# In[91]:


# Let's re-run the model using the selected variables
X_train_sm=sm.add_constant(X_train[col])
lm_3=sm.GLM(y_train,X_train_sm, family=sm.families.Binomial())
res=lm_3.fit()
print(res.summary())


# In[92]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif=pd.DataFrame()
vif["Features"]=X_train[col].columns
z=X_train[col].shape[1]
vif["VIF"]=[variance_inflation_factor(X_train[col].values,i) for i in range(z)]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[93]:


# dropping the column which has high VIF
col=col.drop(['What is your current occupation_Unemployed'],1)


# In[94]:


# rebuild the model
X_train_sm = sm.add_constant(X_train[col])
lm_4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = lm_4.fit()
print(res.summary())


# In[95]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif=pd.DataFrame()
vif["Features"]=X_train[col].columns
z=X_train[col].shape[1]
vif["VIF"]=[variance_inflation_factor(X_train[col].values,i) for i in range(z)]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[96]:


# drop the above two columns from X_train too
drop_col=["Lead Source_Welingak Website",'What is your current occupation_Unemployed']
X_train=X_train.drop(drop_col,axis=1,inplace=True)


# In[97]:


# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[98]:


# Reshape
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# Creating a dataframe with the actual converted lead and the predicted probabilities

# In[99]:


y_train_pred_final=pd.DataFrame({'converted_lead':y_train.values, 'conv_lead_prob':y_train_pred})
y_train_pred_final['lead_No']=y_train.index
y_train_pred_final.head()


# In[100]:


y_train_pred_final["predicted"]=y_train_pred_final['conv_lead_prob'].map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[101]:


confusion = metrics.confusion_matrix(y_train_pred_final.converted_lead, y_train_pred_final.predicted )
print(confusion)


# In[102]:


# Predicted     not_converted    converted
# Actual
# not_converted        3373      458
# converted            700       1659


# In[103]:


#Lets check Accuracy
print("accuracy_score of train data")
print(metrics.accuracy_score(y_train_pred_final["converted_lead"],y_train_pred_final["predicted"]))


# ## Step 8: Metrics Evaluation

# In[104]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[105]:


# Let's calculate the sensitivity of  logistic regression model
print("sensitivity:")
TP / float(TP+FN)


# In[106]:


# Let us calculate specificity
print("specificity:")
TN / float(TN+FP)


# In[107]:


# Calculate false postive rate - predicting churn when customer does not converted lead
print("false postive rate:")
print(FP/ float(TN+FP))


# In[108]:


# positive predictive value 
print("positive predictive value:")
print (TP / float(TP+FP))


# In[109]:


# Negative predictive value
print (TN / float(TN+ FN))


# ## Plotting the ROC Curve
# An ROC curve demonstrates several things:
# 
# It shows the tradeoff between sensitivity and specificity (any increase in sensitivity will be accompanied by a decrease in specificity).
# The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the test.
# The closer the curve comes to the 45-degree diagonal of the ROC space, the less accurate the test.

# In[110]:


# Function to draw_roc
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[111]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.converted_lead, y_train_pred_final.conv_lead_prob, drop_intermediate = False )


# In[112]:


draw_roc(y_train_pred_final.converted_lead, y_train_pred_final.conv_lead_prob)


# ## Step 9: Finding Optimal Cutoff Point

# Optimal cutoff probability is that prob where we get balanced sensitivity and specificity

# In[113]:


numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.conv_lead_prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[114]:


#Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.converted_lead, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[115]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# #### From the curve above, 0.3 is the optimum point to take it as a cutoff probability.

# In[116]:


y_train_pred_final['final_predicted'] = y_train_pred_final.conv_lead_prob.map( lambda x: 1 if x > 0.3 else 0)

y_train_pred_final.head()


# In[117]:


print("accuracy_score with cut_off")
metrics.accuracy_score(y_train_pred_final['converted_lead'], y_train_pred_final['final_predicted'])


# In[118]:


confusion2 = metrics.confusion_matrix(y_train_pred_final['converted_lead'], y_train_pred_final['final_predicted'] )
confusion2


# In[119]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[120]:


# Let's see the sensitivity of  logistic regression model
sensitivity=TP / float(TP+FN)
print("sensitivity=",sensitivity) 


# In[121]:


# Let us calculate specificity
specificity=TN / float(TN+FP)
print("specificity with cut_off=",specificity)


# In[122]:


# Calculate false postive rate - predicting churn when customer does not have churned
print("FPR=",FP/ float(TN+FP))


# In[123]:


# Positive predictive value 
print ("Positive predictive value",TP / float(TP+FP))


# In[124]:


# Negative predictive value
print ("Negative predictive value",(TN / float(TN+ FN)))


# ## Step 10 :Precision and Recall
# 

# In[125]:


#Looking at the confusion matrix again
confusion = metrics.confusion_matrix(y_train_pred_final.converted_lead, y_train_pred_final.predicted )
confusion


# Precision=TP / TP + FP

# In[126]:


Precision=confusion[1,1]/(confusion[0,1]+confusion[1,1])

print("Precision of maodel is :",Precision)   


# Recall=TP / TP + FN

# In[127]:


Recall=confusion[1,1]/(confusion[1,0]+confusion[1,1])
print("Recall of maodel is :",Recall)


# Using sklearn utilities for the same

# In[128]:


from sklearn.metrics import precision_score, recall_score


# In[129]:


precision_score(y_train_pred_final.converted_lead, y_train_pred_final.predicted)


# In[130]:


recall_score(y_train_pred_final.converted_lead, y_train_pred_final.predicted)


# ### Precision and recall tradeoff

# In[131]:


from sklearn.metrics import precision_recall_curve


# In[132]:


y_train_pred_final.converted_lead, y_train_pred_final.predicted


# In[133]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.converted_lead, y_train_pred_final.conv_lead_prob)


# In[134]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# ### Step 11: Making predictions on the test set

# In[135]:


# Scaling vaiables to numeric
X_test[['Total Time Spent on Website','Page Views Per Visit','TotalVisits']] = scaler.transform(X_test[['Total Time Spent on Website',
                                                                                            'TotalVisits','Page Views Per Visit']])


# In[136]:


# Select the columns in X_train for X_test 
X_test = X_test[col]
# Add a constant to X_test
X_test_sm = sm.add_constant(X_test[col])
X_test_sm


# Making predictions on the test set

# In[137]:


# Storing prediction of test set in the variable 'y_test_pred'
y_test_pred = res.predict(X_test_sm)
# Coverting it to df
y_pred_df = pd.DataFrame(y_test_pred)


# In[138]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
# Remove index for both dataframes to append them side by side 
y_pred_df.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[139]:


# Append y_test_df and y_pred_df
y_pred_final = pd.concat([y_test_df, y_pred_df],axis=1)
# Renaming column 
y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})
y_pred_final.head()


# In[140]:


# Making prediction using cut off 0.3
y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.3 else 0)
y_pred_final


# In[141]:


print("Accuracy on test data")
metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted)


# In[142]:


print("confusion matrix")
confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
confusion2


# In[143]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[144]:


print("sensitivity on test data") 
TP/(TP+FN)


# In[145]:


print("specificity on test data")
TN/(TN+FP)


# In[146]:


print("Precision on test data")
TP / (TP + FP)


# In[147]:


print("Recall on test data") 
TP / (TP + FN)


# In[148]:


print("Features used in Final Model :")

print("-----------------------Feature Importance--------------------")
print(res.params)


# ## Conclusion: 
# The logistic regression model predicts the probability of the target variable having a certain value, instead of predicting the value of the target variable directly. Then a cutoff of the probability is used to obtain the predicted value of the target variable.
# 
# Here, the logistic regression model is used to predict the probabilty of conversion of a customer(lead).
# 
# Optimum cut off is chosen to be 0.3 i.e.
# 
# any lead with greater than 0.3 probability of converting is predicted as Hot Lead (customer will convert) and any lead with 0.3 or less probability of converting is predicted as Cold Lead (customer will not convert)

# ## Result on Train data
# 
# Accuracy is 84.21%
# 
# Sensitivity is 83.89%
# 
# Specificity is 84.41%
# 
# Precision : 85.78%
# 
# Recall 78.50%
# 
# 
# ## Result on Test data
# 
# Accuracy : 84.55%
# 
# Sensitivity :85.88%
# 
# Specificity :83.71%
# 
# Precision : 77%
# 
# Recall 85.88%
# 
# Roc : 0.91

# THANK YOU !!
