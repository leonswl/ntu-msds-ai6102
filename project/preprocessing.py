# preprocess data and split training set to train and validation

import pandas as pd
import numpy as np

def convert_bool_to_int(df:pd.DataFrame, col_lst:list) -> pd.DataFrame:
    """
    Simple utility function to convert column from boolean values to integer format. Some ML models are not able to accept boolean values.
    
    Args:
        df [pandas dataframe]: dataframe in pandas format
        col_lst [list]: list of column names in boolean format to be converted to int 
        
    Return:
        df [pandas dataframe]: dataframe in pandas format with boolean columns converted to int
    """

    if not set(col_lst).issubset(set(df.columns)):
        col_lst.remove('Transported')

    # run tests to verify if there are null values in the boolean columns since columns with null values cannot be converted to int
    target_df = df[col_lst] # create new dataframe with only boolean columns
    test_result = any(num > 0 for num in list(target_df.isnull().sum())) # check if column have null values

    # convert boolean to int if there are no null values
    if not test_result: 
        for col in col_lst:
            df[col] = df[col].astype(int)

    else:
        print("There are null values in the columns. Dataframe remains unchanged")

    return df
def preprocessing(df):
    """
    
    """
    num_feats = ['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    # cat_feats = ['HomePlanet','Cabin','Destination','RoomService','Age']

    df[num_feats] = df[num_feats].fillna(value=0)
    df.isnull().sum().sort_values(ascending=False)

    col_lst = ["Transported","VIP","CryoSleep"]

    df = convert_bool_to_int(df, col_lst)

    # split cabin into columns - Deck, Cabin_num and Side
    df["Cabin"].str.split("/", expand=True)
    df[["Deck", "Cabin_num", "Side"]] = df["Cabin"].str.split("/", expand=True)

    # drop Cabin column
    try:
        df = df.drop('Cabin', axis=1)
    except KeyError:
        print("Field does not exist")

    return df

def split_dataset(df, test_ratio=0.20):
  test_indices = np.random.rand(len(df)) < test_ratio
  return df[~test_indices], df[test_indices]

# load input
data_type = int(input("Select data set type to be preprocessed (0 - train, 1 - test): "))


# if data type is train
if data_type == 0:
    # load dataset
    dataset_df = pd.read_csv('data/train.csv')

    # preprocess dataset
    dataset_df = preprocessing(dataset_df)

    # split train dataset into train and validation sets
    train_ds_pd, valid_ds_pd = split_dataset(dataset_df)

    # export datasets to path
    train_ds_pd.to_csv("data/train_ds_pd.csv",index=False)
    valid_ds_pd.to_csv("data/valid_ds_pd.csv",index=False)

    print(
        f"{len(train_ds_pd)} examples in training, {len(valid_ds_pd)} examples in validation."
    )

# if data type is test
elif data_type == 1:
    dataset_df = pd.read_csv('data/test.csv')

    # preprocess dataset
    dataset_df = preprocessing(dataset_df)

    # export datasets to path
    dataset_df.to_csv("data/test_ds_pd.csv",index=False)

    print(
        f"{len(dataset_df)} examples in testing."
    )
    
# invalid data type input
else:
    print("Input is invalid. Please select between '1' or '2'.")





