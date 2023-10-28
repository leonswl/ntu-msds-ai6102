import pandas as pd
import numpy as np

def convert_bool_to_int (df:pd.DataFrame, col_lst:list)->pd.DataFrame:
    """
    Simple utility function to convert column from boolean values to integer format. Some ML models are not able to accept boolean values.
    
    Args:
        df [pandas dataframe]: dataframe in pandas format
        col_lst [list]: list of column names in boolean format to be converted to int 
        
    Return:
        df [pandas dataframe]: dataframe in pandas format with boolean columns converted to int
    """
    
#     run tests to verify if there are null values in the boolean columns since columns with null values cannot be converted to int
    target_df = df[col_lst] # create new dataframe with only boolean columns
    test_result = any(num > 0 for num in list(target_df.isnull().sum())) # check if column have null values
    
    # convert boolean to int if there are no null values
    if test_result == False: 
        for col in col_lst:
            df[col] = df[col].astype(int)
        
    else:
        print("There are null values in the columns. Dataframe remains unchanged")
        
    return df

def split_dataset(dataset, test_ratio=0.20):
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]



# load dataset

dataset_df = pd.read_csv('data//train.csv')

num_feats = ['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_feats = ['HomePlanet','Cabin','Destination','RoomService','Age']

dataset_df[num_feats] = dataset_df[num_feats].fillna(value=0)
dataset_df.isnull().sum().sort_values(ascending=False)

col_lst = ["Transported","VIP","CryoSleep"]

dataset_df = convert_bool_to_int(dataset_df, col_lst)

dataset_df["Cabin"].str.split("/", expand=True)

dataset_df[["Deck", "Cabin_num", "Side"]] = dataset_df["Cabin"].str.split("/", expand=True)

try:
    dataset_df = dataset_df.drop('Cabin', axis=1)
except KeyError:
    print("Field does not exist")

train_ds_pd, valid_ds_pd = split_dataset(dataset_df)
print(
    f"{len(train_ds_pd)} examples in training, {len(valid_ds_pd)} examples in testing."
)