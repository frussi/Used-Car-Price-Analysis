# ------------------------------------------------------------------------
# Program Name: CSC425DetailedDataAnalysisPortfolioProject
# Author: Frank Russo
# Date: 01-15-2023
# ------------------------------------------------------------------------
# Pseudocode: Import data, analyze features, and create decision a tree, KNN, and Naive Bayes model
# ------------------------------------------------------------------------
# Program Inputs: data set of car sales
# Program Outputs: Summary statistics and graphical analasys on dataset.
# Decision tree, KNN model, and Naive Bayes model for predicting if a car is cheap or not
# ------------------------------------------------------------------------

# Import required packages
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from collections import defaultdict
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
import seaborn as sn
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.metrics import accuracy_score as acc

# Import dataset
df = pd.read_csv('CarSales.csv')

# Print Summary Statistics
print(df.describe())
print(df['Brand'].value_counts())
print(df['Body'].value_counts())
print(df['Engine Type'].value_counts())
print(df['Registration'].value_counts())

models = df['Model'].value_counts()
print(models.describe())

print(df['Model'].value_counts())

# Remove outliers
df = df[df['EngineV'] < 10]
dataset = df.astype('category')

# Create categorical variable for price
df.loc[df['Price'] >= 11500, 'PriceC'] = 'Expensive'
df.loc[df['Price'] < 11500, 'PriceC'] = 'Cheap'
df = df.dropna()

# Create and print Histograms
pricesList = list(df['Mileage'])
plt.hist(pricesList, bins=50)
plt.title("Car Mileage Histogram")
plt.show()

fig = sns.displot(df, x="Brand", col="PriceC",binwidth=1, height=3,)
fig = sns.displot(df, x="Body", col="PriceC",binwidth=1, height=3,)
fig = sns.displot(df, x="Mileage", col="PriceC",binwidth=1, height=3,)
fig = sns.displot(df, x="EngineV", col="PriceC",binwidth=1, height=3,)
fig = sns.displot(df, x="Engine Type", col="PriceC",binwidth=1, height=3,)
fig = sns.displot(df, x="Registration", col="PriceC",binwidth=1, height=3,)
fig = sns.displot(df, x="Year", col="PriceC",binwidth=1, height=3,)
plt.show()

print(df['PriceC'].value_counts())

# Preprocess Data
dataset["Brand"] = dataset["Brand"].cat.codes
dataset["Body"] = dataset["Body"].cat.codes
dataset["Engine Type"] = dataset["Engine Type"].cat.codes
dataset["Registration"] = dataset["Registration"].cat.codes
dataset["Model"] = dataset["Model"].cat.codes

dataset = dataset.dropna()

dataset.loc[df['Price'] >= 11500, 'PriceC'] = 1
dataset.loc[df['Price'] < 11500, 'PriceC'] = 0
dataset.drop(df.columns[[1]], axis=1, inplace=True)

features = ['Brand', 'Body', 'Mileage', 'EngineV', 'Engine Type', 'Registration', 'Year', 'Model']

# Create and print correlation matrix
print(dataset.corr(method='pearson', min_periods=1, numeric_only=False))
corr_matrix=dataset.corr(method='pearson', min_periods=1, numeric_only=False)
sn.heatmap(corr_matrix, annot=True)
plt.title("Correlation Matrix of Features")
plt.show()

# Split the dataset into training and testing sets
X = dataset.drop("PriceC", axis=1)
X = X.values
y = dataset['PriceC']
y = y.values

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=0)
print(X_train.shape)
print(X_test.shape)

# Standardize data
st_x= StandardScaler()
X_train= st_x.fit_transform(X_train)
X_test= st_x.transform(X_test)

# Create Decision Tree
print("Decision Tree:")

# Create Decision Tree Model
dtree = DecisionTreeClassifier(random_state=0)
dtree.fit(X_train,y_train)
y_train_pred = dtree.predict(X_train)
y_test_pred = dtree.predict(X_test)

# Print Accuracy of Decision Tree
print(f'Train Score {accuracy_score(y_train_pred,y_train)}')
print(f'Test Score {accuracy_score(y_test_pred,y_test)}')

# Plot Decision Tree
tree.plot_tree(dtree, feature_names=features)
plt.title("Decision Tree for Car Price")
plt.show()

# Prune Decision Tree with cost complexity pruning
path = dtree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
print(ccp_alphas.shape)

# Add model to a list for each alpha
dtrees = []
for ccp_alpha in ccp_alphas:
    dtree = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    dtree.fit(X_train, y_train)
    dtrees.append(dtree)

# Plot accuracy vs alpha graph and print
train_acc = []
test_acc = []
for c in dtrees:
    y_train_pred = c.predict(X_train)
    y_test_pred = c.predict(X_test)
    train_acc.append(accuracy_score(y_train_pred,y_train))
    test_acc.append(accuracy_score(y_test_pred, y_test))

plt.scatter(ccp_alphas, train_acc)
plt.scatter(ccp_alphas, test_acc)
plt.plot(ccp_alphas,train_acc,label='train_accuracy', drawstyle="steps-post")
plt.plot(ccp_alphas,test_acc,label='test_accuracy', drawstyle="steps-post")
plt.legend()
plt.title('Decision Tree Accuracy vs Alpha')
plt.show()

# Create pruned decision tree model
dtree_ = DecisionTreeClassifier(random_state=0,ccp_alpha=0.005)
dtree_.fit(X_train,y_train)
y_train_pred = dtree_.predict(X_train)
y_pred = dtree_.predict(X_test)

# Print Accuracy of pruned Decision Tree
print(f'Train Score {accuracy_score(y_train_pred,y_train)}')
print(f'Test Score {accuracy_score(y_pred,y_test)}')

# Print pruned Decision Tree
tree.plot_tree(dtree_, feature_names=features)
plt.title("Pruned Decision Tree for Car Price")
plt.show()

# Print decision tree accuracy, metrics, and feature importance
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
print(dtree_.feature_importances_)

# Print decision tree confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True)
plt.title("Decision Tree Confusion Matrix")
plt.show()

# Print decision tree ROC curve
y_pred_proba = dtree_.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title("Decision Tree ROC Curve")
plt.show()
auc = metrics.roc_auc_score(y_test, y_pred_proba)
print(auc)

# Create KNN model
print("KNN:")

# Create list of error rates for n = 1 - 40
error_rate = []
for i in range(1,40):
 knn = KNeighborsClassifier(n_neighbors=i, metric='manhattan')
 knn.fit(X_train, y_train)
 pred_i = knn.predict(X_test)
 error_rate.append(np.mean(pred_i != y_test))

# Plot error rates vs neighbors
plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('KNN Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
req_k_value = error_rate.index(min(error_rate)) + 1
print("Minimum error:-", min(error_rate), "at K =", req_k_value)
plt.show()

# Build and fit KNN model
knn = KNeighborsClassifier(n_neighbors = req_k_value, metric='manhattan')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(knn)

# Print KNN accuracy, metrics
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Print KNN confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True)
plt.title("KNN Confusion Matrix")
plt.show()

# Print KNN ROC curve
y_pred_proba = knn.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title("KNN ROC Curve")
plt.show()
auc = metrics.roc_auc_score(y_test, y_pred_proba)
print(auc)

# Create Naive Bayes classifier
print("Naive Bayes:")
naiveB = GaussianNB()

# Create the forward sequential feature selection
sfs1 = sfs(naiveB,
           k_features=3,
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=5)

# Use forward sequential feature selector to select features
sfs1 = sfs1.fit(X_train, y_train)

# Print results of selected features
feat_cols = list(sfs1.k_feature_idx_)
print(feat_cols)

# Build model using all features
naiveB = GaussianNB()
naiveB.fit(X_train, y_train)

# Print training and testing accuracy of all features
y_train_pred = naiveB.predict(X_train)
print('Training accuracy on all features: %.3f' % acc(y_train, y_train_pred))

y_test_pred = naiveB.predict(X_test)
print('Testing accuracy on all features: %.3f' % acc(y_test, y_test_pred))

# Build model with selected features
naiveB = GaussianNB()
naiveB.fit(X_train[:, feat_cols], y_train)

# Print training and testing accuracy of selected features
y_train_pred = naiveB.predict(X_train[:, feat_cols])
print('Training accuracy on selected features: %.3f' % acc(y_train, y_train_pred))

y_test_pred = naiveB.predict(X_test[:, feat_cols])
print('Testing accuracy on selected features: %.3f' % acc(y_test, y_test_pred))

y_pred = naiveB.predict(X_test[:, feat_cols])
print(naiveB)

# Print Naive Bayes accuracy, metrics
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
print(metrics.classification_report(y_test, y_pred))

# Print Naive Bayes confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True)
plt.title("Naive Bayes Confusion Matrix")
plt.show()

# Print Naive Bayes ROC curve
y_pred_proba = naiveB.predict_proba(X_test[:, feat_cols])[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title("Naive Bayes ROC Curve")
plt.show()
auc = metrics.roc_auc_score(y_test, y_pred_proba)
print(auc)