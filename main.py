import pandas as pd
import warnings
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

# *** 1. Loading dataset ***
dataset = pd.read_csv("survey.csv")
print(dataset.shape)

# *** 2. Cleaning our dataset ***
# Delete some of the features that we don't want to use.
dataset = dataset.drop(["Timestamp"], axis=1)
dataset = dataset.drop(["state"], axis=1)
dataset = dataset.drop(["comments"], axis=1)
dataset = dataset.drop(["Country"], axis=1)
dataset = dataset.drop(["tech_company"], axis=1)
dataset = dataset.drop(["obs_consequence"], axis=1)

# Changing missing (NA) values with TEMP.
dataset = dataset.fillna("TEMP")
'''print(dataset.iloc[0])'''

# Changing 'self_employed' values with 'No'.
dataset["self_employed"].replace(["TEMP"], "No", inplace=True)
'''print(dataset["self_employed"])
print(dataset["self_employed"].unique())'''

# Changing 'work_interfere' with 'Don't know'
dataset["work_interfere"].replace(["TEMP"], "Don't know", inplace=True)

# Changing Age values with median if age < 18 or > 120.
median = int(dataset["Age"].median())
print(median)

for i in range(dataset.shape[0]):
    if dataset["Age"][i] < 18 or dataset["Age"][i] > 120:
        dataset["Age"][i] = median

# Creating age_range feature.
dataset['age_range'] = pd.cut(dataset['Age'], [0, 20, 30, 65, 100], labels=["0-20", "21-30", "31-65", "66-100"],
                              include_lowest=True)

# Creating 3 arrays for genders (male, female, trans).
male = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man", "msle", "mail", "malr",
        "cis man", "cis male", "p"]
female = ["cis female", "f", "female", "woman", "femake", "female ", "cis-female/femme", "female (cis)", "femail",
          "a little about you"]
trans = ["trans-female", "something kinda male?", "queer/she/they", "non-binary", "nah", "all", "enby", "fluid",
         "genderqueer",
         "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter",
         "female (trans)", "queer", "ostensibly male, unsure what that really means"]

dataset["Gender"] = dataset["Gender"].str.lower()

# Decreasing gender values to male, female, trans.
for i in range(dataset.shape[0]):
    if dataset["Gender"][i] in male:
        dataset["Gender"][i] = "male"

    elif dataset["Gender"][i] in female:
        dataset["Gender"][i] = "female"

    elif dataset["Gender"][i] in trans:
        dataset["Gender"][i] = "trans"

# *** 3. Encoding dataset ***
# Backup dataset beforec encoding for graphs.
backupDataset = dataset.copy()

# Encoding dataset with LabelEncoder.
for i in dataset:
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(dataset[i])
    dataset[i] = label_encoder.transform(dataset[i])

# Scaling age.
ageScaler = MinMaxScaler()
dataset['Age'] = ageScaler.fit_transform(dataset[['Age']])

# Checking whether there are any missing values or not.
print(dataset.isnull().sum())

# *** 4. Plotting charts for dataset visualization ***
# Distribution by age.
mu = backupDataset["Age"].mean()
sigma = backupDataset["Age"].std()
num_bins = backupDataset["Age"].unique().size

fig, ax = plt.subplots()
n, bins, patches = ax.hist(backupDataset["Age"], num_bins, density=True)
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
ax.plot(bins, y, '--')
ax.set_xlabel('Age')
ax.set_title("Distribution by Age")
fig.tight_layout()

# Distribution by gender.
trans = 0
female = 0
male = 0
for i in backupDataset["Gender"]:
    if i == "trans":
        trans = trans + 1
    elif i == "female":
        female = female + 1
    else:
        male = male + 1

fig1, ax1 = plt.subplots()
ax1.pie([male, female, trans], labels=["male", "female", "trans"], autopct='%1.1f%%')
plt.title("Distribution by gender")

# Mental health condition by family history and gender.
sns.catplot(x="family_history", y="treatment", hue="Gender", data=dataset, kind="bar").set_xticklabels(["NO", "YES"])
plt.title('Mental Health Condition by Family History And Gender')
plt.xlabel('Family History')

# Mental health condition by agerange.
sns.catplot(x="age_range", y="treatment", hue="Gender", data=dataset, kind="bar", ci=None).set_xticklabels(
    ["0-20", "21-30", "31-65", "66-100"])
plt.title('Mental Health Condition by Age Range')
plt.xlabel('Age range')

# *** 5. Determining feature importance ***
# Defining X and y for feature importance.
X_importance = dataset.drop(["treatment"], axis=1)
y_importance = dataset["treatment"]

# ExtraTreesClassifier for computing feature importance.
ETC = ExtraTreesClassifier(n_estimators=300, random_state=0).fit(X_importance, y_importance)
importance = ETC.feature_importances_
std = np.std([tree.feature_importances_ for tree in ETC.estimators_], axis=0)
indices = np.argsort(importance)[::-1]

labels = []
for col in dataset.columns:
    if col == "treatment":
        continue
    labels.append(col)

plt.figure(figsize=(12, 8))
plt.title("Feature importance's")
plt.bar(range(21), importance[indices], color="blue", yerr=std[indices], align="center")
plt.xticks(range(21), labels, rotation='vertical')
plt.xlim([-1, X_importance.shape[1]])
plt.show()

# Defining X and y.
dataset_features = ["Age", "Gender", "self_employed", "family_history", "work_interfere", "no_employees", "remote_work",
                    "benefits", "care_options"]
X = dataset[dataset_features]
y = dataset["treatment"]

# Split X and y for training and testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
