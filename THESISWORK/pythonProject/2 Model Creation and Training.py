import pandas
# Imported Libraries

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import time
import xgboost as xgb
# Classifier Libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

# Other Libraries
from sklearn.preprocessing import StandardScaler  # for preprocessing the data
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, auc, roc_curve, classification_report
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

#-------------------------------------------------------------------------------------------------------------
#                                                       Functions
#Function for plotting ROC_AUC curve
def plot_roc_auc(y_test, preds):
    '''
    Takes actual and predicted(probabilities) as input and plots the Receiver
    Operating Characteristic (ROC) curve
    '''
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def runKNN(X_train, X_test, y_train, y_test,returnDataset):
    #                                                   K-Neighbours Classifier
    print("Starting KNN..")
    start = time.time()
    knn.fit(X_train, y_train)
    stop = time.time()
    print(f"Training time: {stop - start}s")
    y_pred = knn.predict(X_test)
    resulted_confussion_matrix = confusion_matrix(y_test, y_pred)

    np.set_printoptions(precision=2)
    disp= plot_confusion_matrix(knn, X_test,y_test,display_labels=["Non Fraud","Fraudulent"], cmap=plt.cm.Blues,colorbar=False)
    disp.ax_.set_title("KNN Algorithm")
    plt.show()

    print("Classification Report for K-Nearest Neighbours: \n", classification_report(y_test, y_pred))
    print("Confusion Matrix of K-Nearest Neigbours: \n", resulted_confussion_matrix)
    plot_roc_auc(y_test, knn.predict_proba(X_test)[:, 1])
    if returnDataset == 1:
        return resulted_confussion_matrix, falsePositiveDataset(X_test, y_test, y_pred)
    else:
        return resulted_confussion_matrix

def XGBoost(X_train, X_test, y_train, y_test,returnDataset):
    #                                           XGBoost Classifier
    print("XGBoost Classifier..")
    start = time.time()
    XGBoost_CLF.fit(X_train, y_train)
    stop = time.time()
    print(f"Training time: {stop - start}s")

    y_pred = XGBoost_CLF.predict(X_test)
    resulted_confussion_matrix = confusion_matrix(y_test, y_pred)

    np.set_printoptions(precision=2)
    disp = plot_confusion_matrix(XGBoost_CLF, X_test,y_test,display_labels=["Non Fraud","Fraudulent"], cmap=plt.cm.Blues,colorbar=False)
    disp.ax_.set_title("XGBoost Algorithm")
    plt.show()

    print("Classification Report for XGBoost: \n", classification_report(y_test, y_pred))
    print("Confusion Matrix of XGBoost: \n", resulted_confussion_matrix)
    plot_roc_auc(y_test, XGBoost_CLF.predict_proba(X_test)[:, 1])
    if returnDataset == 1:
        return resulted_confussion_matrix, falsePositiveDataset(X_test, y_test, y_pred)
    else:
        return resulted_confussion_matrix

def RandomForest(X_train, X_test, y_train, y_test,returnDataset):
    #                                           Random Forest
    print("Starting Random Forest..")
    start = time.time()
    rf_clf.fit(X_train, y_train)
    stop = time.time()
    print(f"Training time: {stop - start}s")

    y_pred = rf_clf.predict(X_test)
    resulted_confussion_matrix = confusion_matrix(y_test, y_pred)

    np.set_printoptions(precision=2)
    disp = plot_confusion_matrix(rf_clf, X_test,y_test,display_labels=["Non Fraud","Fraudulent"], cmap=plt.cm.Blues,colorbar=False)
    disp.ax_.set_title("Random Forest Algorithm")
    plt.show()

    print("Classification Report for Random Forest Classifier: \n", classification_report(y_test, y_pred))
    print("Confusion Matrix of Random Forest Classifier: \n", resulted_confussion_matrix)
    plot_roc_auc(y_test, rf_clf.predict_proba(X_test)[:, 1])
    if returnDataset == 1:
        return resulted_confussion_matrix, falsePositiveDataset(X_test, y_test, y_pred)
    else:
        return resulted_confussion_matrix


def combinedKNNRF(X_train, X_test, y_train, y_test,returnDataset):
    #                                           Combined Classifiers
    print("KNN RF..")
    estimators = [("KNN", knn), ("rf", rf_clf)]
    ens = VotingClassifier(estimators=estimators, voting="soft", weights=[1, 2])
    start = time.time()
    ens.fit(X_train, y_train)
    stop = time.time()
    print(f"Training time: {stop - start}s")

    y_pred = ens.predict(X_test)
    resulted_confussion_matrix = confusion_matrix(y_test, y_pred)

    np.set_printoptions(precision=2)
    disp = plot_confusion_matrix(ens, X_test,y_test,display_labels=["Non Fraud","Fraudulent"], cmap=plt.cm.Blues,colorbar=False)
    disp.ax_.set_title("Combined KNN and RF")
    plt.show()

    print("Classification Report for Ensembled Models: \n",
          classification_report(y_test, y_pred))
    print("Confusion Matrix of Ensembled Models: \n", resulted_confussion_matrix)
    plot_roc_auc(y_test, ens.predict_proba(X_test)[:, 1])
    if returnDataset == 1:
        return resulted_confussion_matrix, falsePositiveDataset(X_test, y_test, y_pred)
    else:
        return resulted_confussion_matrix


def combinedAll(X_train, X_test, y_train, y_test,returnDataset):
    print("KNN RF XGB..")
    estimators = [("KNN", knn), ("rf", rf_clf), ("xgb", XGBoost_CLF)]
    ens = VotingClassifier(estimators=estimators, voting="soft", weights=[1, 4, 1])
    start = time.time()
    ens.fit(X_train, y_train)
    stop = time.time()
    print(f"Training time: {stop - start}s")

    y_pred = ens.predict(X_test)
    resulted_confussion_matrix = confusion_matrix(y_test, y_pred)

    np.set_printoptions(precision=2)
    disp = plot_confusion_matrix(ens, X_test,y_test,display_labels=["Non Fraud","Fraudulent"], cmap=plt.cm.Blues,colorbar=False)
    disp.ax_.set_title("Combined Algorithms")
    plt.show()


    print("Classification Report for Ensembled Models: \n",
          classification_report(y_test, y_pred))
    print("Confusion Matrix of Ensembled Models: \n", resulted_confussion_matrix)
    plot_roc_auc(y_test, ens.predict_proba(X_test)[:, 1])
    if returnDataset == 1:
        return resulted_confussion_matrix,falsePositiveDataset(X_test,y_test,y_pred)
    else:
        return resulted_confussion_matrix


#                                   Retrieving the false positives and storing them in excel
def falsePositiveDataset(X_test,y_test,y_pred):
    #RETURNING WHETHER THE ORIGINAL RESULTS WHETHER FRAUD OR NOT
    print("y_test.head()")
    print(type(y_test))
    print(y_test.head())

    #STORING A DATAFRAME OF ALL
    try:
        listofNonFraud= y_test.to_frame().T
        listofNonFraud = listofNonFraud.T
    except:
        listofNonFraud = y_test

    #Keeping those that where not fraudulent
    listofNonFraudAccurate = listofNonFraud[listofNonFraud["fraud"] != 1]
    listofNonFraudAccurate = listofNonFraudAccurate.sort_index()

    #gets all predictions
    y_predDataframe = pd.DataFrame(list(y_pred), columns=["fraud"])

    #keeps all predictions which where fraudulent
    y_predDataframePositive = y_predDataframe.loc[y_predDataframe['fraud'] == 1]
    y_predDataframePositive = y_predDataframePositive.sort_index()


    #Returns the full columns of dataset where detected as positive
    allDataPositivePredictions = pd.DataFrame(columns=["customer", "age", "merchant", "category", "Amount"])
    for index, value in y_predDataframePositive.iterrows():
        allDataPositivePredictions = allDataPositivePredictions.append(X_test.iloc[index, :])

    #Here the records which index of the predicted as fraud are matched with those which actual resulted as not fraud
    falsePositveDataframe = listofNonFraudAccurate[listofNonFraudAccurate.index.isin(allDataPositivePredictions.index)]
    print("falsePositveDataframe")
    print(falsePositveDataframe.head())
    print(len(falsePositveDataframe))

    print("All records which where detected as false positive")
    print(data_r)

    #used previous index and got full dataset with non modified values
    fullDataframe = data_r[data_r.index.isin(falsePositveDataframe.index)]
    print(fullDataframe)
    print(fullDataframe.size)
    return fullDataframe


#-------------------------------------------------------------------------------------------------------------
#                                               Retrieving the dataset

#dataset
#https://www.kaggle.com/ntnu-testimon/banksim1?select=bs140513_032310.csv

data = pandas.read_csv("dataset/bs140513_032310.csv")
print("Database Accessed")
print(data.info())

# Taking all the fraud records
df_fraud = data.loc[data['fraud'] == 1]

# Taking the same number of records of the majority class(No Frauds)
df_non_fraud = data.loc[data['fraud'] == 0]

#-------------------------------------------------------------------------------------------------------------
#                                                Data Cleaning
print("Unique zipCodeOri values: ", data.zipcodeOri.nunique())
print("Unique zipMerchant values: ", data.zipMerchant.nunique())
# dropping zipcodeOri and zipMerchant since they have only one unique value

data_r = data.copy()
data_r = data_r.drop(['zipcodeOri', 'zipMerchant', 'step', 'gender'], axis=1)

print(data_r)

#This is created inorder to retain original values within the dataset for false positive analysis
data_reduced = data_r.copy()

# since it will take a lot of time to train
# turning object columns type to categorical for easing the transformation process to numeric
col_categorical = data_reduced.select_dtypes(include=['object']).columns
for col in col_categorical:
    data_reduced[col] = data_reduced[col].astype('category')

# categorical values ==> numeric values
data_reduced[col_categorical] = data_reduced[col_categorical].apply(lambda x: x.cat.codes)

#mormalised records
data_reduced['Amount'] = StandardScaler().fit_transform(data_reduced['amount'].values.reshape(-1, 1))
data_reduced = data_reduced.drop(['amount'], axis=1)

print("data_reduced")
print(data_reduced)

#-------------------------------------------------------------------------------------------------------------
#                                       Find the correlation using a heatmap
corr=data_reduced.corr()
sns.heatmap(corr, annot = True)
plt.show()
#Highest correlation was found from amount and secondly found to be the category which was negatively correlation

# -------------------------------------------------------------------------------------------------------------
#                                                   Normal Dataset
X = data_reduced.drop(['fraud'], axis=1)
y = data['fraud']
print(X.head(), "\n")
print(y.head())
# This shows the total number of fraudulent transactions
print(y[y == 1].count())

# ------------------------------------------------------------------------------------------------------------
#                                                       SMOTE

# to produce the same results across a different run  -- SMOTE(random_state=42)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
y_res = pd.DataFrame(y_res)
print(y_res)
print(X_res)

# cross validation wasn't applied since we have a lot of instances
# it should be better to cross validate most of the times

print("Base accuracy score we must beat is: ",df_non_fraud.fraud.count() / np.add(df_non_fraud.fraud.count(), df_fraud.fraud.count()) * 100)
# Our accuracy is very high but we are not detecting any frauds so it is a useless classifier.

# ------------------------------------------------------------------------------------------------------------
#                                             Initialization of models
# initialisation of knn
knn = KNeighborsClassifier(n_neighbors=5, p=1)
#n_neighbors is setting 5 to k
#p was set to 1 inorder to use the manhattan_distance

# initialisation of Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42,
                                verbose=1, class_weight="balanced")
#n_estimator was set to 100 to create a 100 trees within the forest classifier
#it was set to a 100 and not more since that it would increase the time complexity if increased
#max depth was set to 8 meaning it is the longest path between the root node and the leaf node meaning levels
#so 8 levels under the root node
#bootstrap are set as default to true
#random state takes control of the randomness of bootstrapping when building the trees and sampling of the features when looking for the best split at each node
#verbose set to 1 inorder to get the red logs with elapsed time , not changed to 2 since it would print the rest of the information about building each tree
#class_weight was set to balanced since it uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data



XGBoost_CLF = xgb.XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=400,
                                objective="binary:hinge", booster='gbtree',
                                n_jobs=-1, nthread=None, gamma=0,random_state=42)
#model_depth 6 layers also usually the default value
#learning_rate After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.
#n_estimator set to 400 referring to the number of trees to be used in the XGBoost
#objective was set to "binary:hinge" this makes predictions of 0 or 1, rather than producing probabilities.
#booster Can be gbtree, gblinear or dart; gbtree and dart use tree based models while gblinear uses linear functions.
#tree based models allow to represent all types of non-linear data well, since no formula is needed which describes the relation between target and input variables.
#n_jobs is set to -1 which references the number of parallel threads used to run xgboost which uses all cores
#nthread is set to None which references the number of threads available
#gamma set to default 0 the higher it is the fewer the leaves
#random_state is set to 42 and it is a parameter used for initializing the internal random number generator, which will decide the splitting of data into train and test indices

# ------------------------------------------------------------------------------------------------------------
#                                                   RUN ALL FUNCTIONS
testSize = [0.2, 0.33, 0.5]
# random state = is used by setting a number and from it if 2 seperate splits are done than they would be split exactly the same
# stratify is used inorder to keep the orginal proportions. It will be provided with the y array and both the training and test will have the same proportion


#if dataset of false positives is needed returnDataset needs to be 1 otherwise set as 0
returnDataset = 0

#Normal Dataset Training
for x in testSize:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=x, random_state=42, shuffle=True, stratify=y)
    runKNN(X_train, X_test, y_train, y_test,returnDataset)
    RandomForest(X_train, X_test, y_train, y_test,returnDataset)
    XGBoost(X_train, X_test, y_train, y_test,returnDataset)
    combinedKNNRF(X_train, X_test, y_train, y_test,returnDataset)
    combinedAll(X_train, X_test, y_train, y_test,returnDataset)


print("Starting SMOTE Training")
#SMOTE Training
for x in testSize:
    X_train, X_test, y_train, y_test = train_test_split(X_res,y_res,test_size=x,random_state=42,shuffle=True,stratify=y_res)
    runKNN(X_train, X_test, y_train, y_test,returnDataset)
    RandomForest(X_train, X_test, y_train, y_test,returnDataset)
    XGBoost(X_train, X_test, y_train, y_test,returnDataset)
    combinedKNNRF(X_train, X_test, y_train, y_test,returnDataset)
    combinedAll(X_train, X_test, y_train, y_test,returnDataset)

# ------------------------------------------------------------------------------------------------------------
#                                           Saves the chosen false positive records to excel
#Here we obtain the best results selected from the excel sheet and save the false positives to the Excel Sheet

# These are the best outcomes according to previously run code
returnDataset = 1

#                                                   Normal Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

# returning confusion matrix and dataframe of false positve records
cm1,returnedDataframe = XGBoost(X_train, X_test, y_train, y_test,returnDataset)

print(cm1[0])
print("Test")
print(returnedDataframe.head())

# determining the name of the file
file_name1 = 'falsePositiveNormalDataset.csv'
# saving the excel
returnedDataframe.to_csv("dataset/"+file_name1, sep=',', encoding='utf-8', index=False)
print('DataFrame is written to Excel File successfully.')


#                                                   SMOTE Training
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, shuffle=True, stratify=y_res)

cm2,returnedDataframe2 = runKNN(X_train, X_test, y_train, y_test,returnDataset)
print(cm2[0])
print(returnedDataframe2.head())


#Creating Excel File
# determining the name of the file
file_name2 = 'falsePositiveOversampledDataset.csv'
# saving the excel
returnedDataframe2.to_csv("dataset/"+file_name2, sep=',', encoding='utf-8', index=False)
print('DataFrame is written to Excel File successfully.')

#------------------------------------------------------------------------------------------------------------