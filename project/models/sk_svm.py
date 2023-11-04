import os
import logging
import datetime
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score,  recall_score, precision_score
import pandas as pd

def load_datasets(data_path):
    """
    Utility function to load datasets based on given path. Training, validation and test sets must be generated from the preprocessing script.
    
    Args:
        data_pathj [str]: data path where train, validation and test sets are stored

    Returns
        train_df, valid_df, test_df [pandas dataframe]: loaded datasets using pandas dataframe
    """
    logging.info(os.path.dirname(__file__))
    data_path = os.path.join(os.path.dirname(__file__), data_path)

    ## load train and validation dataset
    train_df = pd.read_csv(os.path.join(data_path, "train_ds_pd.csv"))
    valid_df = pd.read_csv(os.path.join(data_path, "valid_ds_pd.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test_ds_pd.csv"))
    logging.info("Succesfully loaded training, validation and test datasets")

    return train_df, valid_df, test_df

class SVM:
    def __init__(self, train_df:pd.DataFrame, valid_df:pd.DataFrame, test_df:pd.DataFrame, label:str):
        logging.info("============ Instantiating SVM class ============")
        # self.train_df = pd.concat([train_df,valid_df])
        self.train_df = train_df
        self.X_train = self.train_df.drop([label],axis=1).values
        self.y_train = self.train_df[[label]].values

        self.valid_df = valid_df
        self.X_valid = self.valid_df.drop([label],axis=1).values
        self.y_valid = self.valid_df[[label]].values

        self.test_df = test_df
        self.X_test = self.test_df.values
        self.label = label
        self.submission_id = test_df.PassengerId
        logging.info("*********** SVM class successfully instantiated *********** ")

    def feature_selection(self, selected_features:list=None):
        """
        Function to update training, validation and test data with selected features
        """
        logging.info("============ Pruning Features ============")
        self.selected_features = selected_features

        # update datasets
        self.train_df = self.train_df[selected_features]
        selected_features.remove(self.label)
        self.test_df = self.test_df[selected_features]

        logging.info("*********** Feature selection completed successfully *********** ")

    def prepare_data(self,encoder:str="OneHotEncoder"):
        """
        Function to encode categorical features
        
        Args:
            encoder [str]: Optional; Determines the type of encoding used for encoding categorical features. Default='OneHotEncoder'. Options include OneHotEncoder, OrdinalEncoder
        """
        logging.info("============ Preparing data using {%s} ============", encoder)
        if encoder == "OneHotEncoder":
            enc = OneHotEncoder(categories="auto",handle_unknown='ignore')
            
        elif encoder == 'OrdinalEncoder':
            enc = OrdinalEncoder(categories="auto",handle_unknown='ignore')
            
        # fit using training set
        enc.fit(X=self.X_train)

        # transform X_train
        X_train_transformed = enc.transform(X=self.X_train)

        # transform X_valid
        X_valid_transformed = enc.transform(X=self.X_valid)

        # transform X_test
        X_test_transformed = enc.transform(X=self.X_test)

        self.X_train = X_train_transformed.toarray()
        self.X_valid = X_valid_transformed.toarray()
        self.X_test = X_test_transformed.toarray()

        logging.info("X_train array has the following shape: {%s}", self.X_train.shape)
        logging.info("X_test array has the following shape: {%s}", self.X_test.shape)

        logging.info("*********** Data Preparation completed successfully *********** ")

    def __GridSearch(self, kernels, gammas,c):
        """Function to perform GridSearch to obtain optimal hyperparameters"""
        logging.info("============ Executing GridSearch for model hyperparameters ============")
        clf = svm.SVC()
        clf.fit(self.X_train,self.y_train)
        param_grid = dict(kernel=kernels, C=c, gamma=gammas)
        grid = GridSearchCV(
            clf, 
            param_grid, 
            cv=10, 
            n_jobs=-1, 
            verbose=3,
            scoring='accuracy')
        grid.fit(self.X_train, self.y_train)
        best_params = grid.best_params_

        logging.info("*********** GridSearchCV completed successfully")
        
        return best_params.get('C'), best_params.get('gamma'), best_params.get('kernel')
    
    def create_svm_model(self, cv:bool=False, kernels:list=None, c:list=None, gammas:list=None):
        """
        Function to create SVM model. Cross validation is an optional parameter

        Args:
            cv [bool]: True/False to use cross validation. optional, default is True. 
        
        """
        logging.info("============ Creating SVM model ============")
        if not cv:
            self._model = svm.SVC()
        else:
            best_C, best_gamma, best_kernel = self.__GridSearch(kernels, c, gammas)
            self._model = svm.SVC(
                kernel=best_kernel,
                gamma=best_gamma,
                C=best_C)
            logging.info("Optimal SVM hyperparameters - C: {%.4f}; gamma: {%.4f}, kernel: {%s}", best_C, best_gamma, best_kernel)
            
    def run_experiments(self):
        logging.info("============ Running Experiment ============")
        self._model.fit(self.X_train, self.y_train)
        logging.info("*********** Experiment completed successfully ***********")

    def evaluate(self):
        """Function to evaluate model using validation sets and return a set of metrics"""
        logging.info("============ Evaluating Model ============")
        y_pred= self._model.predict(self.X_valid)

        self.accuracy_score = accuracy_score(y_true=self.y_valid, y_pred=y_pred)
        self.f1_score = f1_score(y_true=self.y_valid, y_pred=y_pred)
        self.recall_score = recall_score(y_true=self.y_valid, y_pred=y_pred)
        self.precision_score = precision_score(y_true=self.y_valid, y_pred=y_pred)

        logging.info("*********** Model Evaluation completed successfully ***********")

        return self.accuracy_score, self.f1_score, self.recall_score, self.precision_score

    def predict(self):
        """
        Function to make predictions
        
        Returns:
            predictions: 
            output [pandas dataframe]: submission output pd dataframe

        """
        logging.info("============ Generating Predictions ============ ")
        self.predictions = self._model.predict(self.X_test)
        self.output = pd.DataFrame({
            'PassengerId': self.submission_id,
            'Transported': self.predictions
            })
        
        logging.info("*********** Predictions completed successfully. Output file for submission has been generated and exported to desired path")
        
        return self.predictions, self.output

def sk_svm():
    # get current timestamp
    current_time = datetime.datetime.now()

    # configure logging
    log_filename = f"logs/svm/svm_output_{current_time}" 
    os.makedirs(os.path.dirname(log_filename),exist_ok=True)
    file_handler = logging.FileHandler(log_filename, mode="w", encoding=None, delay=False)
    logging.basicConfig(filename=log_filename, level=logging.INFO)

    # load datasets
    train_df, valid_df, test_df = load_datasets(data_path="../data")
    
    # set parameters
    select_features = ['CryoSleep','Age','RoomService','Cabin_num','FoodCourt', 'ShoppingMall', 'Spa', 'HomePlanet', 'Side', 'Deck', 'Transported', 'VRDeck','Destination']
    logging.info("Feature selection will be executed over the following selected features: {%s}", select_features)

    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    # c = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 1e2, 1e3, 1e4, 1e5]
    c = [0.1, 1, 10]
    gammas = [0.1, 1, 10]
    logging.info("GridSearch configurations - kernels: {%s}; c: {%s}; gammas: {%s}", kernels, c, gammas)

    # experiment SVM models
    clf = SVM(train_df=train_df,valid_df=valid_df, test_df=test_df,label='Transported')
    clf.feature_selection(selected_features=select_features)
    clf.prepare_data(encoder="OneHotEncoder")
    clf.create_svm_model(cv=True,kernels=kernels, c=c, gammas=gammas)
    clf.run_experiments()
    # evaluate model using validation set and get model metrics
    accuracy_score, f1_score, recall_score, precision_score = clf.evaluate()
    logging.info("Accuracy score is {%.4f}; F1 score is {%.4f}; Recall score is {%.4f}; Precision score is {%.4f}", accuracy_score, f1_score, recall_score, precision_score)

    # predict test data
    predictions, output = clf.predict()

    # export output to path
    output_path = os.path.join(os.path.dirname(__file__),"..","submissions")
    os.makedirs(output_path, exist_ok=True) 
    output.to_csv(os.path.join(output_path,"sk_svm.csv"),index=False)

if __name__ == "__main__":
    sk_svm()


