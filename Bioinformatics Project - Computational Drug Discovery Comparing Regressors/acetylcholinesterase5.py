import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import lazypredict
from lazypredict.Supervised import LazyRegressor
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns


# Read csv file
df = pd.read_csv("acetylcholinesterase_06_bioactivity_data_3class_pIC50_pubchem_fp.csv")

X = df.drop('pIC50', axis=1)
Y = df.pIC50

# Data pre-processing
# print(X.shape)
selection = VarianceThreshold(threshold=(.8 * (1 - .8)))    
X = selection.fit_transform(X)
# print(X.shape)

# Perform data splitting using 80/20 ratio
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# Compare ML algorithms, defines and builds the lazyclassifier
clf = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None)         # LazyRegressor is a class from lazypredict library. It automatically trains and evaluates a variety of regression models without requiring you to explicitly define them.
models_train,predictions_train = clf.fit(X_train, X_train, Y_train, Y_train)
models_test,predictions_test = clf.fit(X_train, X_test, Y_train, Y_test)

# Performance table of the training set (80% subset)
# print(predictions_train)


# Performance table of the test set (20% subset)
# print(predictions_test)


# Data visualization of model performance

# # Bar plot of R-squared values
# #train["R-Squared"] = [0 if i < 0 else i for i in train.iloc[:,0] ]

# plt.figure(figsize=(10, 15))
# sns.set_theme(style="whitegrid")
# ax = sns.barplot(y=predictions_train.index, x="R-Squared", data=predictions_train)
# print(ax.set(xlim=(0, 1)))
# plt.show()


# # Bar plot of RMSE values
# plt.figure(figsize=(5, 10))
# sns.set_theme(style="whitegrid")
# ax = sns.barplot(y=predictions_train.index, x="RMSE", data=predictions_train)
# print(ax.set(xlim=(0, 10)))
# plt.show()


# Bar plot of calculation time
plt.figure(figsize=(5, 10))
sns.set_theme(style="whitegrid")
ax = sns.barplot(y=predictions_train.index, x="Time Taken", data=predictions_train)
print(ax.set(xlim=(0, 10)))
plt.show()