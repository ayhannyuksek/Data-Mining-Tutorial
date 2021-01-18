import pandas as pd
import warnings
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

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
# print(dataset.isnull().sum())

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


#  6. Applying algorithms

# Function for plotting confusion matrix
def confusionMatrix(y_test, prediction, modelName):
    plt.figure(figsize=(12, 8))
    mat = confusion_matrix(y_test, prediction)
    sns.heatmap(mat, square=True, annot=True, cbar=True, fmt="d")
    plt.xlabel("Pred")
    plt.ylabel("Real Value")
    plt.title("Confusion Matrix of " + modelName)
    plt.show()


# Defining X and y.
dataset_features = ["Age", "Gender", "self_employed", "family_history", "work_interfere", "no_employees", "remote_work",
                    "benefits", "care_options"]
X = dataset[dataset_features]
y = dataset["treatment"]

# Split X and y for training and testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Logistic Regression part
print("********* Logistic Regression *********")
log = LogisticRegression().fit(X_train, y_train)
log_pred = log.predict(X_test)
log_accuracy = accuracy_score(y_test, log_pred)
print("Accuracy score: ", log_accuracy)
print("Real value: ", y_test.values[:30], "\nPred value: ", log_pred[:30])
# confusionMatrix(y_test, log_pred, "Logistic Regression")

# AdaBoost Classifier part
print("\n********* AdaBoost Classifier *********")
ABC = AdaBoostClassifier(n_estimators=50).fit(X_train, y_train)
ABC_pred = ABC.predict(X_test)
ABC_accuracy = accuracy_score(y_test, ABC_pred)
print("Accuracy score: ", ABC_accuracy)
print("Real value: ", y_test.values[:30], "\nPred value: ", ABC_pred[:30])
# confusionMatrix(y_test, ABC_pred, "AdaBoost Classifier")

# KNeighbors Classifier part
print("\n********* KNeighbors Classifier *********")
n_values = {}
for i in range(1, 53):
    KNC = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
    KNC_pred = KNC.predict(X_test)
    KNC_accuracy = accuracy_score(y_test, KNC_pred)
    n_values[i] = KNC_accuracy

best_n_value = max(n_values, key=n_values.get)
KNC = KNeighborsClassifier(n_neighbors=best_n_value).fit(X_train, y_train)
KNC_pred = KNC.predict(X_test)
KNC_accuracy = accuracy_score(y_test, KNC_pred)
print("Accuracy score: ", KNC_accuracy)
print("Real value: ", y_test.values[:30], "\nPred value: ", KNC_pred[:30])
# confusionMatrix(y_test, KNC_pred, "KNeighbors Classifier")

# Random Forest Classifier part
print("\n********* Random Forest Classifier *********")
depth_values = {}
for i in range(1, 11):
    RFC = RandomForestClassifier(max_depth=i).fit(X_train, y_train)
    RFC_pred = RFC.predict(X_test)
    RFC_accuracy = accuracy_score(y_test, RFC_pred)
    depth_values[i] = RFC_accuracy

best_depth_value = max(depth_values, key=depth_values.get)
RFC = RandomForestClassifier(max_depth=best_depth_value).fit(X_train, y_train)
RFC_pred = RFC.predict(X_test)
RFC_accuracy = accuracy_score(y_test, RFC_pred)
print("Accuracy score: ", RFC_accuracy)
print("Real value: ", y_test.values[:30], "\nPred value: ", RFC_pred[:30])
# confusionMatrix(y_test, RFC_pred, "Random Forest Classifier")

# Bagging Classifier part
print("\n********* Bagging Classifier *********")
node_values = {}
for i in [500, 2000, 8000, 99999]:
    DTC = DecisionTreeClassifier(max_leaf_nodes=i)
    bag = BaggingClassifier(base_estimator=DTC).fit(X_train, y_train)
    bag_pred = bag.predict(X_test)
    bag_accuracy = accuracy_score(y_test, bag_pred)
    node_values[i] = bag_accuracy

best_node_value = max(node_values, key=node_values.get)
DTC = DecisionTreeClassifier(max_leaf_nodes=best_node_value)
bag = BaggingClassifier(base_estimator=DTC).fit(X_train, y_train)
bag_pred = bag.predict(X_test)
bag_accuracy = accuracy_score(y_test, bag_pred)
print("Accuracy score: ", bag_accuracy)
print("Real value: ", y_test.values[:30], "\nPred value: ", bag_pred[:30])
# confusionMatrix(y_test, bag_pred, "Bagging Classifier")

# 7. Comparison Graph of Algorithms by Accuracy Score
fig, ax = plt.subplots(figsize=(16, 9))
print(" ")
models = ["Logistic Regression", "AdaBoost Classifier", "KNeighbors Classifier", "Random Forest Classifier", "Bagging Classifier"]
success = [log_accuracy * 100, ABC_accuracy * 100, KNC_accuracy * 100, RFC_accuracy * 100, bag_accuracy * 100]
ax.bar(models, success)
ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)
for i in range(0, 5):
    print(models[i] + ":", success[i])


