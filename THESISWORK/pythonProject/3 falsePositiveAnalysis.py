# importing the module
import pandas
import matplotlib.pyplot as plt

import pandas as pd

normalDB=pandas.read_csv("dataset/falsePositiveNormalDataset.csv")
overSampledDB=pandas.read_csv("dataset/falsePositiveOversampledDataset.csv")

print(normalDB.head())
print(overSampledDB.head())

#age
#labels=["2: 26-35","3: 36-45","4: 46:55","1: 19-25","6: > 65","0: <= 18","5: 56:65"]
labels=["26-35","36-45","46:55","19-25","> 65","<= 18","56:65"]
plot = normalDB['age'].value_counts().plot(kind='pie',autopct='%1.0f%%',labeldistance=None)
plt.title('Age Ratio for false positives')
plot.set_ylabel('')
plt.legend(labels, loc="best")
plt.show()

#merchant
plot =normalDB['merchant'].value_counts().plot(kind='pie',autopct='%1.0f%%',labeldistance=None)
plt.title('Merchant Ratio for false positives')
plot.set_ylabel('')
plt.legend(bbox_to_anchor=(1,0.5), loc="center right", fontsize=10,
           bbox_transform=plt.gcf().transFigure)
plt.subplots_adjust(left=0.0, bottom=0.1, right=0.60)
plt.show()

#category
plot =normalDB['category'].value_counts().plot(kind='pie',autopct='%1.0f%%',labeldistance=None)
title=plt.title('Category Ratio for false positives')
plot.set_ylabel('')
plt.legend(bbox_to_anchor=(1,0.5), loc="best", fontsize=10,
           bbox_transform=plt.gcf().transFigure)
plt.subplots_adjust(left=0.0, bottom=0.1, right=0.60)
plt.show()

#amount
labels=['0-100','100-200','200-300','300-400','400-500','500-10000']
normalDB['bins'] = pd.cut(normalDB['amount'], bins=[0,100,200,300,400,500,10000], labels=labels, right=True)
bin_percent = pd.DataFrame(normalDB['bins'].value_counts(normalize=True) * 100)
plot = bin_percent.plot.pie(y='bins', figsize=(8, 8), autopct='%1.1f%%',labeldistance=None)
plt.title('Amount Range for False Positive Records')
plot.set_ylabel('')
plt.show()

#all amounts in dataframe
ax = normalDB.plot.bar(x='bins',y='amount', rot=0)
ax.set_xlabel("")
ax.axes.get_xaxis().set_visible(False)
plt.show()

#indicates the amount for a specific merchant was declared as false positive
normalDB.plot(x ='amount', y='merchant', kind = 'scatter')
plt.show()

#Grouping by columns for understanding every merchant and age
g= normalDB.groupby("age")
print("Grouped up records according to age")
for age, ageDF in g:
    print("The age column is "+age)
    print(ageDF)

merchant= normalDB.groupby("merchant")
print("Grouped up records according to the merchant")
for merchant, merchantDF in merchant:
    print("The merchant column is "+merchant)
    print(merchantDF)
    #Draws bar chart for every amount per merchant
    # ax = merchantDF.plot.bar(x='customer', y='amount', rot=0,stacked=True)
    # plt.title('Merchant of Id ='+merchant+'had Customer Purchases for false positives')
    # plt.show()

c= normalDB.groupby("category")
print("Grouped up records according to category")
for category, categoryDF in c:
    print("The Category column is "+category)
    print(categoryDF)

#------------------------------------------------------------------------------------------------------------
#                                               Same analysis but for the oversampled dataset
print("Starting analysis for SMOTE dataset")
#age
#labels=["2: 26-35","3: 36-45","4: 46:55","1: 19-25","6: > 65","0: <= 18","5: 56:65"]
labels=["26-35","36-45","46:55","19-25","> 65","56:65","<= 18","U:Unknown"]
plot = overSampledDB['age'].value_counts().plot(kind='pie',autopct='%1.0f%%',labeldistance=None)
plt.title('Age Ratio for false positives')
plot.set_ylabel('')
plt.legend(labels, loc="best")
plt.show()

#merchant
plot =overSampledDB['merchant'].value_counts().plot(kind='pie',autopct='%1.0f%%',labeldistance=None)
plt.title('Merchant Ratio for false positives')
plot.set_ylabel('')
plt.legend(bbox_to_anchor=(1,0.5), loc="center right", fontsize=10,
           bbox_transform=plt.gcf().transFigure)
plt.subplots_adjust(left=0.0, bottom=0.1, right=0.60)
plt.show()

#category
plot =overSampledDB['category'].value_counts().plot(kind='pie',autopct='%1.0f%%',labeldistance=None)
title=plt.title('Category Ratio for false positives')
plot.set_ylabel('')
plt.legend(bbox_to_anchor=(1,0.5), loc="best", fontsize=10,
           bbox_transform=plt.gcf().transFigure)
plt.subplots_adjust(left=0.0, bottom=0.1, right=0.60)
plt.show()

#amount
labels=['0-100','100-200','200-300','300-400','400-500','500-10000']
overSampledDB['bins'] = pd.cut(overSampledDB['amount'], bins=[0,100,200,300,400,500,10000], labels=labels, right=True)
bin_percent = pd.DataFrame(overSampledDB['bins'].value_counts(normalize=True) * 100)
plot = bin_percent.plot.pie(y='bins', figsize=(8, 8), autopct='%1.1f%%',labeldistance=None)
plt.title('Amount Range for False Positive Records')
plot.set_ylabel('')
plt.show()

#all amounts in dataframe
ax = overSampledDB.plot.bar(x='bins',y='amount', rot=0)
ax.set_xlabel("")
ax.axes.get_xaxis().set_visible(False)
plt.show()

#indicates the amount for a specific merchant was declared as false positive
overSampledDB.plot(x ='amount', y='merchant', kind = 'scatter')
plt.show()

#Grouping by columns for understanding every merchant and age
g= overSampledDB.groupby("age")
print("Grouped up records according to age")
for age, ageDF in g:
    print("The age column is "+age)
    print(ageDF)

merchant= overSampledDB.groupby("merchant")
print("Grouped up records according to the merchant")
for merchant, merchantDF in merchant:
    print("The merchant column is "+merchant)
    print(merchantDF)

c= overSampledDB.groupby("category")
print("Grouped up records according to category")
for category, categoryDF in c:
    print("The Category column is "+category)
    print(categoryDF)