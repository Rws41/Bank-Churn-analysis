import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from IPython.display import display
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix



# I downloaded this data set from Kaggle. Thanks to user Marslino Edward.
# The goal of this will be to predict customer bank churn.

#Setting working directory to be location of file
os.chdir(os.path.dirname(os.path.abspath(__file__)))


#Loading the data and getting some information about the data.
df = pd.read_csv("./data/bank_churn.csv")
n = df.shape[0]
df.info()
print(df.isna().sum())

display(df)



#Exploratory Data Analysis
#Checking the demographics of bank customers.
fig, axs = plt.subplots(ncols = 3, figsize=(8,5))
sns.countplot(data=df, x='Geography', hue='Geography', legend=None, ax=axs[0]).set_ylabel(None)
sns.countplot(data=df, x='Gender', hue='Gender', legend=None, ax=axs[1]).set_ylabel(None)
sns.histplot(data=df, x='Age', bins=9, ax=axs[2]).set_ylabel(None)
fig.suptitle("Demographics of Bank Customers")
fig.supylabel("Count of Customers")
fig.tight_layout()
plt.show()
plt.close()
#So most customers are men and in their 30's and 40's and French.



#Now lets check information about their financial information
fig, axs = plt.subplots(ncols = 4, figsize=(12,5))
sns.histplot(data=df, x='CreditScore', bins = 5, legend=None, ax=axs[0]).set_ylabel(None)
sns.histplot(data=df, x='EstimatedSalary', bins=6, legend=None, ax=axs[1]).set_ylabel(None)
sns.histplot(data=df, x='Tenure', bins=10, ax=axs[2]).set_ylabel(None)
sns.histplot(data=df, x='Balance', ax=axs[3]).set_ylabel(None)
fig.suptitle("Financial Information of Bank Customers")
fig.supylabel("Count of Customers")
fig.tight_layout()
plt.show()
plt.close()

#Looking at this most people have credit scores in the around ~600 - ~700 
#We also have a unifrom salary distribution going from 0 to 200,000.
#Good representation of people's tenure at the bank. 
# Balance appears to be relatively normal with  a mean ~120,000, aside from a huge number of customers with 0 balance.


#Next, lets evaluate bank engagement
fig, axs = plt.subplots(ncols = 3, figsize=(8,5))
sns.countplot(data=df, x='HasCrCard', hue='HasCrCard', legend=None, ax=axs[0]).set_ylabel(None)
sns.countplot(data=df, x='IsActiveMember', hue='IsActiveMember', legend=None, ax=axs[1]).set_ylabel(None)
sns.countplot(data=df, x='NumOfProducts', ax=axs[2]).set_ylabel(None)

axs[0].set_xticklabels(['No', 'Yes'])
axs[1].set_xticklabels(['No', 'Yes'])
fig.suptitle("Engagement of Bank Customers with Products")
fig.supylabel("Count of Customers")
fig.tight_layout()
plt.show()
plt.close()

#Finally lets evaluate some of these and determine what churn looks like for some of these demographics
churned = df.loc[df['Exited'] == 1]
print("Percent of customers who churn: ", (churned.shape[0] /n)*100)


#Adjusting columns for labels
churned.loc[:, 'HasCrCard'] = churned['HasCrCard'].astype(str).apply(lambda x: 'Has Card' if x == '1' else 'No Card')
churned.loc[:, 'IsActiveMember'] = churned['IsActiveMember'].astype(str).apply(lambda x: 'Is Active' if x== '1' else 'Is Inactive')
cat_cols = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
fix, axs = plt.subplots(nrows= 1, ncols = 4, figsize=(15,10))


#Making Pie charts to examine churn
for i in range(len(cat_cols)):
    exited = churned[cat_cols[i]].value_counts()
    axs[i].pie(exited, autopct = '%0.2f%%', labels = exited)
    axs[i].legend(exited.index)

plt.tight_layout()
plt.show()
plt.close()

#So looks like in our data set, fewer spaniards churn, women churn more, those who have a card churn, and inactive members churn.



#There are some columns that are irrelevant for prediction purposes. Lets remove those
to_drop = ['RowNumber', 'CustomerId', 'Surname']
df.drop(columns=to_drop, inplace=True)

#I will also generate a dataframe without Geography and compare how the inclusion impacts things
df_no_location = df.drop(columns=['Geography'])

#Factorizing Geography - Categories are Germany, Spain and France.
#Also factorizing 'Gender" Column which has the sexes as strings
df = pd.get_dummies(data=df, columns=['Geography'])
df['Gender'] = df['Gender'].apply(lambda x: 1 if x=='Male' else 0)

#Getting the Correlation Matrix to evaluate for valuable predictors and multicollinearity.
corr_matrix = df.corr()
plt.figure()
sns.heatmap(corr_matrix, annot=True, linewidths=0.2, fmt=".1f")

#Based on the correlation matrix we can see most items are not correlated
#There are only some mild correlations between predictors and the response variable with the highest being 0.3
#The only moderate correlation is a correlation between Balance and being from Germany.


#Data preparation
#Shuffle the data
df = shuffle(df)

#Pulling out the response variable
y = df['Exited']
df.drop(labels = "Exited", inplace= True, axis=1)



#Split the data
x_train, x_test, y_train, y_test = train_test_split(df, y, train_size=0.7)



#Standardize the data
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#Put all the data into a container for convenience
split_data = [x_train, x_test, y_train, y_test]

#Prediction. We will evaluate a few models.
svm = SVC()
randfor = RandomForestClassifier()
knn = KNeighborsClassifier()
ada = AdaBoostClassifier()



#Generating parameters for evaluation
learning_rates = np.arange(0.01, 1.07, 0.05)
svc_C = np.arange(0.5, 1.6, 0.5)
tree_count = np.arange(50, 301, 25)
neighbors = np.arange(2, 21, 2)

svc_parameters = {'kernel':('linear','rbf'), 'C':svc_C}
forest_parameters = {'n_estimators': tree_count}
knn_parameters = {'n_neighbors': neighbors}
ada_parameters = {'learning_rate': learning_rates, 'algorithm': ['SAMME']}


#putting models and parameters in list for evaluation
models = [svm, randfor, knn, ada]
parameters = [svc_parameters, forest_parameters, knn_parameters, ada_parameters]



#Function to carry out the fitting, predicting, scoring, and reporting  accuracy of models
def model_predict(model, parameters, data, best):
    xtrain, xtest, ytrain, ytest = data[0], data[1], data[2], data[3]
    grid_searcher = GridSearchCV(model, parameters)
    grid_searcher.fit(xtrain, ytrain)
    ypred = grid_searcher.predict(xtest)
    accuracy = grid_searcher.score(xtest, ytest)


    #Got accuracy get model and score
    if accuracy > best[1]:
        best = (model, accuracy)

    #Create nice graphic of performance
    con_mat = confusion_matrix(ytest, ypred)

    plt.figure()
    sns.heatmap(con_mat, annot=True)
    plt.title("{} = {}%".format(model, round(accuracy*100, 2)))
    plt.show()
    plt.close()
    return best




#initialize and run through all the models.
best_score = (None, 0)
for i in range(len(models)):
    best_score = model_predict(models[i], parameters[i], split_data, best_score)


#Final Results
print("Based on this run, the best model is ", best_score[0], "with an accuracy of ", [best_score[1]])



