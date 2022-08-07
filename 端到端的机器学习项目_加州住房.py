import os
import tarfile
import urllib.request

DOWLOAD_ROOT="https://github.com/ageron/handson-ml2/raw/master/datasets/housing/"
HOUSEING_PATH=os.path.join("datasets","housing")
HOUSEING_URL=DOWLOAD_ROOT+"datasets/housing/housing.tgz"

def fetch_housing_data(housing_usl=HOUSEING_URL,housing_path=HOUSEING_PATH):
    os.makedirs(housing_path,exist_ok=True)
    tgz_path=os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(housing_usl,tgz_path)
    housing_tgz=tarfile.open(tgz_path)
   
#fetch_housing_data()

import pandas as pd
def load_husing_data(housing_path=HOUSEING_PATH):
    csv_path= os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)

housing =load_husing_data();
print(housing.head())
print(housing.info())

print(housing["ocean_proximity"].value_counts())

print(housing.describe())

import  matplotlib as mpt
housing.hist(bins=50,figsize=(20,15))
#mpt.pyplot.show()

import numpy as np

def spt_train_test(data,test_ratio):
    shuffled_indices=np.random.permutation(len(data))
    test_set_size= int(len(data))*test_ratio
    test_indices=shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

train_set,test_set=spt_train_test(housing,0.2)
print(len(train_set))
print(len(test_set))
