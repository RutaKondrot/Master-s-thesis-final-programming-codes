import re
import string
import time
import pandas as pd
import numpy as np
# pd.options.display.max_rows = None
pd.options.display.max_columns = None
import gc
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns


df_job = pd.read_csv('C:\\Users\\el ruchenzo\\jobsproject\\jobsproject\\cleaned_data_adj1.csv', encoding='utf-8', usecols=['cleaned_description', 'code'], dtype={'code': 'str'})
# print(df_job.dtypes)
# df_job = df_job.dropna()
df_job = df_job.drop_duplicates()
# print(len(df_job)) #7226
# print(df_job['code'].nunique()) #637

# Classificator's code counts:
code_counts = df_job.groupby('code').size()
code_counts = code_counts.sort_values(ascending=False)
# code_counts.to_csv("value_counts.csv", encoding='utf-8')
print(code_counts)

# Removing classificator's codes which have only 1 job description
df_job = df_job[df_job['code'].map(df_job['code'].value_counts()) > 1]
# ValueError: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2. (this appears in the TRAIN-TEST SPLIT part)
# print(len(df_job)) #7047
# print(df_job['code'].nunique()) #458


##########
# TF-IDF #
##########

docs = list(df_job['cleaned_description'])
tfidf_vectorizer = TfidfVectorizer(use_idf=True)#, max_features = 25000) # True - enables the use of the IDF weighting; there is no default size for max_features
tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(docs)
print(f"Number of features: {tfidf_vectorizer_vectors.shape[1]}") # in this dataset there are 24021 features
docs = tfidf_vectorizer_vectors.toarray()

X = docs
y = df_job['code']
print(X.shape, y.shape)

SEED = 123

### Initial model's paramters:
mnb = MultinomialNB()
svc = LinearSVC(class_weight='balanced')
dt = DecisionTreeClassifier(random_state=SEED)
rf = RandomForestClassifier(random_state=SEED)
knn = KNeighborsClassifier()

columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Support']
df_results = pd.DataFrame(columns=columns)

start_time = time.time()

df_proportions = pd.Series(df_job['code']).value_counts(normalize=True)
print("Proportions in dataset:")
print(df_proportions)

#Print distribution of job descriptions
# fig = go.Figure([
#     go.Bar(
#         x=df_job['code'].value_counts().index.astype(str),
#         y=df_job['code'].value_counts().tolist(),
#         text=df_job['code'].value_counts().tolist(),
#         textposition='auto',
#         textfont=dict(size=12),
#         marker=dict(color='black')
#     )
# ])
#
# # Update the layout
# fig.update_layout(
#     title = "Classificator's codes of the National Version of ISCO",
#     template="plotly_white",  # Use the white theme
#     # title="The Dataset's Distribution",#"Number of Job Descriptions According to the National Version of ISCO",
#     title_x=0.5,
#     xaxis_title="Codes of the National Version of ISCO",
#     yaxis_title="Number of Job Descriptions",
#     width=1000,  # Width in pixels
#     height=600,  # Height in pixels
# )

# fig.show()

# Split dataset into train (70%) and test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
train_proportions = pd.Series(y_train).value_counts(normalize=True)
print("Proportions in y_train:")
print(train_proportions)
test_proportions = pd.Series(y_test).value_counts(normalize=True)
print("Proportions in y_test:")
print(test_proportions)

# Check if shapes of X_train/test and y_train/test match
print(len(y_test), len(X_test), len(y_train), len(X_train))
 
###
### MNB:
###
mnb.fit(X_train, y_train)

y_pred_train = mnb.predict(X_train)
y_pred_test = mnb.predict(X_test)
report = classification_report(y_test, y_pred_test, zero_division=0, output_dict=True)
print(classification_report(y_test, y_pred_test, zero_division=0))
# model results by classificator's codes
# report_df = pd.DataFrame(report).transpose()
# report_df.to_csv("classification_report_mnb.csv", index=True)

weighted_avg = report['weighted avg']
accuracy = accuracy_score(y_test, y_pred_test)

# print(accuracy)
# print(report)
print(classification_report(y_test, y_pred_test, zero_division=0))
# print(weighted_avg)
mnb_results = {
    'Model': 'MNB',
    'Accuracy': accuracy,
    'Precision': weighted_avg['precision'],
    'Recall': weighted_avg['recall'],
    'F1-score': weighted_avg['f1-score'],
    'Support': weighted_avg['support']
}

mnb_results_df = pd.DataFrame([mnb_results])
print(f"Initial MNB results: ")
# print(mnb_results_df)

# Append the dictionary to the DataFrame
df_results = pd.concat([df_results, mnb_results_df], ignore_index=True)
# print(df_results)

# probs = mnb.predict_proba(X_test)
# preds = probs[:,1]
# fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
# roc_auc = metrics.auc(fpr, tpr)
#
# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()


###
### SVM:
###

svc.fit(X_train, y_train)

y_pred_train = svc.predict(X_train)
y_pred_test = svc.predict(X_test)
report = classification_report(y_test, y_pred_test, zero_division=0, output_dict=True)
# model results by classificator's codes
# report_df = pd.DataFrame(report).transpose()
# report_df.to_csv("classification_report_svm.csv", index=True)
weighted_avg = report['weighted avg']
accuracy = accuracy_score(y_test, y_pred_test)
# print(report)
# print(weighted_avg)

svm_results = {
    'Model': 'SVM',
    'Accuracy': accuracy,
    'Precision': weighted_avg['precision'],
    'Recall': weighted_avg['recall'],
    'F1-score': weighted_avg['f1-score'],
    'Support': weighted_avg['support']
}

svm_results_df = pd.DataFrame([svm_results])
print("Initial SVM results: ")
# print(svm_results_df)
print(classification_report(y_test, y_pred_test, zero_division=0))
# Append the dictionary to the DataFrame
df_results = pd.concat([df_results, svm_results_df], ignore_index=True)
# print(df_results)

###
### DT:
###

dt.fit(X_train, y_train)

y_pred_train = dt.predict(X_train)
y_pred_test = dt.predict(X_test)

report = classification_report(y_test, y_pred_test, zero_division=0, output_dict=True)
# model results by classificator's codes
# report_df = pd.DataFrame(report).transpose()
# report_df.to_csv("classification_report_dt.csv", index=True)

weighted_avg = report['weighted avg']
accuracy = accuracy_score(y_test, y_pred_test)
# print(report)
# print(weighted_avg)

dt_results = {
    'Model': 'DT',
    'Accuracy': accuracy,
    'Precision': weighted_avg['precision'],
    'Recall': weighted_avg['recall'],
    'F1-score': weighted_avg['f1-score'],
    'Support': weighted_avg['support']
}

dt_results_df = pd.DataFrame([dt_results])
print("Initial DT results: ")
# print(dt_results_df)
print(report)

# Append the dictionary to the DataFrame
df_results = pd.concat([df_results, dt_results_df], ignore_index=True)
# print(df_results)

##
## RF:
##

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
# model results by classificator's codes
# report_df = pd.DataFrame(report).transpose()
# report_df.to_csv("classification_report_rf.csv", index=True)

weighted_avg = report['weighted avg']
# accuracy = accuracy_score(y_test, y_pred_test)
# from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred, zero_division=0))
# print(weighted_avg)

rf_results = {
    'Model': 'RF',
    'Accuracy': accuracy,
    'Precision': weighted_avg['precision'],
    'Recall': weighted_avg['recall'],
    'F1-score': weighted_avg['f1-score'],
    'Support': weighted_avg['support']
}

rf_results_df = pd.DataFrame([rf_results])
print("Initial  RF results: ")
# print(rf_results_df)
print(report)

# Append the dictionary to the DataFrame
df_results = pd.concat([df_results, rf_results_df], ignore_index=True)
# print(df_results)

###
### KNN:
###

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
# model results by classificator's codes
# report_df = pd.DataFrame(report).transpose()
# report_df.to_csv("classification_report_knn.csv", index=True)

weighted_avg = report['weighted avg']
accuracy = accuracy_score(y_test, y_pred)
# print(report)
# print(weighted_avg)

knn_results = {
    'Model': 'KNN',
    'Accuracy': accuracy,
    'Precision': weighted_avg['precision'],
    'Recall': weighted_avg['recall'],
    'F1-score': weighted_avg['f1-score'],
    'Support': weighted_avg['support']
}

knn_results_df = pd.DataFrame([knn_results])
print("Initial KNN results: ")
# print(knn_results_df)
print(classification_report(y_test, y_pred, zero_division=0))
# Append the dictionary to the DataFrame
df_results = pd.concat([df_results, knn_results_df], ignore_index=True)
# print(df_results)

print("All models were finished")

df_results = df_results.sort_values(by='Model')  # , ascending=[False, True])
df_results[['Accuracy', 'Precision', 'Recall', 'F1-score']] = df_results[
    ['Accuracy', 'Precision', 'Recall', 'F1-score'] ].mul(100).round(1)
print(df_results)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
