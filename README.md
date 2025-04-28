# homework11
K means Clustering

from sklearn.cluster import KMeans

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline

# read file into dataframe
filepath = "Mall_Customers.csv"

df = pd.read_csv(filepath)
df.head()

# remove CustomerID column
df.drop('CustomerID', axis=1, inplace=True)

# verify updated column names
df.head()

# number of customers by gender
sns.countplot(df['Gender'])
plt.show()


# distribution of customer age
sns.displot(df['Age'])

# distribution of customer income
sns.displot(df['Income'])
plt.show()

# distribution of customer spending score
sns.displot(df['Score'])
plt.show()

# function to assign age groups
def age_groups(age):
    if age <= 25:
        group = '18-25'
    elif age <= 40:
        group = '26-40'
    elif age < 60:
        group = '40-59'
    elif age >= 60:
        group = '60+'
        
    return group

    # use function on "Age" column
df['age_groups'] = df['Age'].apply(age_groups)

# function to assign income groups
def income_groups(income):
    if income < 35:
        group = 'under $35K'
    elif income < 65:
        group = '$35-64K'
    elif income < 100:
        group = '$65-100K'
    elif income >= 100:
        group = '$100K+'
        
    return group

    # use function on "Income" column
df['income_groups'] = df['Income'].apply(income_groups)

# verify changes to dataframe
df.sample(5)

# mean average spending by gender
pd.pivot_table(data=df, index='Gender', values='Score')

# mean average spending score by age group
pd.pivot_table(data= df, index=['age_groups'], values=['Score'])

# mean average spending score by income group
pd.pivot_table(data= df, index=['income_groups'], values=['Score'])

index_val = ['Gender', 'age_groups', 'income_groups']

# mean average spending score by gender, age group, and income group
pd.pivot_table(data= df, index=index_val, values=['Score'])

 compare all numerical features by income group
sns.pairplot(data=df, hue="income_groups")

# dataframe that will be used in algorithm
X = df[['Age', 'Income', 'Score']]
X.head()

# initialize KMeans to create 5 clusters
kmeans = KMeans(n_clusters=5)

# build the model 
# determine centroid position, then assign data to groups based on closest centroid
kmeans.fit(X)

len(df)

# centroid row position DOES NOT MATTER
# centroid columns are in order of dataframe columns index(0=Age, 1=Income, 2=Score)
kmeans.cluster_centers_

# create column in original dataframe with cluster group number
df['cluster'] = kmeans.labels_
df.sample(5)

# compare all numerical features by cluster group
sns.pairplot(data=df, hue='cluster')



plt.show()
