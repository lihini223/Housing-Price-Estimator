import os
import tarfile
import hashlib
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


#function to fetch housing data from the source
def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


fetch_housing_data()


#funtion to load housing data
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()

#displays first 5 rows of the data frame

#displays description of the data. [total no of rows, tyepe, no of non-null values]
# housing.info()
#print(housing.head())
#group by distinct values and their count
# housing["ocean_proximity"].value_counts()

#shows the summary of a numerical attributes
# housing.describe()

#plotting histograms
# housing.hist(bins=50, figsize=(20,15))
# plt.show()

#creating test set
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
#print(len(train_set), "train +", len(test_set), "test") 


#function to check whether it shold go to test set or not
def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash,))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index() #adds an index column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")


#creating income category
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    split_train_set = housing.loc[train_index]
    split_test_set = housing.loc[test_index]

#print(housing["income_cat"].value_counts() / len(housing))


#removing income category attribute
for set in (split_train_set, split_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)


#create a copy of training dataset
housing = split_train_set.copy()


#creating a scatterplot of geographical data
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1,
                            s=housing["population"]/100, label="population",
                            c="median_house_value", cmap=plt.get_cmap("jet"),
                            colorbar=True,
                            )
# plt.show() 
# plt.legend()


corr_matrix = housing.corr()

#print(corr_matrix["median_house_value"].sort_values(ascending=False))


#plotting correlation between attributes using pandas scatter_matrix()

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
# plt.show() 

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)



# prepare the data for machine learning algorithms

housing = split_train_set.drop("median_house_value", axis=1)
housing_labels = split_train_set["median_house_value"].copy()

# Data cleaning 
housing.dropna(subset=["total_bedrooms"]) #option 01 - removes rows and columns with null values
housing.drop("total_bedrooms", axis=1)#option 02 
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median)#option 03

imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)

imputer.fit(housing_num)

# print(imputer.statistics_) 
# print(housing_num.median())

# transform the training set by replacing the missing values by learned medians
X = imputer.transform(housing_num)

# trnasforming the numpy array containing the transformed features intp a pandas dataframe
housing_tr = pd.DataFrame(X, columns=housing_num.columns)


#handling text and categorical attributes
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)

# print(housing_cat_encoded)

# print(encoder.classes_) ['<1H OCEAN' 'INLAND' 'ISLAND' 'NEAR BAY' 'NEAR OCEAN'] [0 = <1H OCEAN]

encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1)) # outputs a scipy sparse matrix
# print(housing_cat_1hot.toarray()) convert the sparse matrix into an array

#short method to convert text categories to integer categories and from integer categories to one-hot categoties
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)
# print(housing_cat_1hot) gives a numpy array by default
#custom transformers
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        
        attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
        housing_extra_attribs = attr_adder.transform(housing.values)
