# Imported Libraries
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

data = pandas.read_csv("dataset/bs140513_032310.csv")
print("")
print(data.info())
#this dataset has 4112 customers

columnNames = []
# iterating the columns
for col in data.columns:
    print(col)
    columnNames.append(col)

print(data.head())
print("")

print('No. of fraud transactions: {}, No. of non-fraud transactions: {}'.format((data.fraud == 1).sum(), (data.fraud == 0).sum()))
print("")

#categorized ages with fraud
print("Ages")
AgeInfo = (data.groupby('age')['fraud'].mean()*100).reset_index().sort_values(by='age' , ascending = False).rename(columns={'fraud':'fraud_percent'})
print(AgeInfo)
#Age: Categorized age
# 0: <= 18,
# 1: 19-25,
# 2: 26-35,
# 3: 36-45,
# 4: 46:55,
# 5: 56:65,
# 6: > 65
# U: Unknown
#fraud occurs more in ages equal and below 18(0th category) whereas the 4th category proved to be the secondly most fraudulent transactions
print("")

print("Zip Code")
print("Unique zipCodeOri values: ",data.zipcodeOri.nunique())
print("Unique zipMerchant values: ",data.zipMerchant.nunique())
print("")

print("Genders")
print(data.loc[data.fraud == 1,'gender'].value_counts())
print("")

print("Categories")
print(data.loc[data.fraud == 1,'category'].value_counts())
print("")

print("Amounts for fraud and nonfraud")
print('Min, Max amount of fraud transactions {}, {}'.format(data.loc[data.fraud == 1].amount.min(), data.loc[data.fraud == 1].amount.max()))
print('Min, Max amount of genuine transactions {}, {}'.format(data.loc[data.fraud == 0].amount.min(), data.loc[data.fraud == 0].amount.max()))
print("")

print("Mean feature values per category",data.groupby('category')['amount','fraud'].mean())
#leisure and the travel is the most selected categories by fraudsters
print("")


#Average amount spend for categories are similar; between 0-500 discarding the outliers, except for the travel category which goes very high
# Plot histograms of the amounts in fraud and non-fraud data
plt.figure(figsize=(30,10))
sns.boxplot(x=data.category,y=data.amount)
plt.title("Boxplot for the Amount spend in category")
plt.ylim(0,4000)
plt.legend()
plt.show()

# Taking all the minority class
df_fraud = data.loc[data['fraud'] == 1]

# Taking the majority class (Non Fraudulent Transactions)
df_non_fraud = data.loc[data['fraud'] == 0]

# Plots a histograms on the amounts for fraud and non-fraud data
plt.hist(df_fraud.amount, alpha=0.5, label='fraud',bins=100)
plt.hist(df_non_fraud.amount, alpha=0.5, label='nonfraud',bins=100)
plt.title("Histogram for fraudulent and nonfraudulent payments")
plt.ylim(0,10000)
plt.xlim(0,1000)
plt.legend()
plt.show()
#fradulent transactions are less in count but more in amount
print("")

print("Are there null values in the dataset?")
print(data.isnull().values.any())
print("")

trans_by_cust = data[data['customer'] == "'C1978250683'"]
fraud_by_cust = data[(data['customer'] == "'C1978250683'") & (data['fraud'] == 1)]
no_fraud_by_cust = data[(data['customer'] == "'C1978250683'") & (data['fraud'] == 0)]
trans_by_cust.head()
num_fraud_trans, num_safe_trans, total_trans = len(fraud_by_cust), len(no_fraud_by_cust), len(trans_by_cust)
percent_frauds = (num_fraud_trans/total_trans * 100)
percent_safe = (100 - percent_frauds)
print("Percentage of frauds by customer C1978250683: ", round(percent_frauds, 2))
print("Percentage of  no frauds by customer C1978250683: ", round(percent_safe, 2))
#High percentage of fraud transactions 41%

