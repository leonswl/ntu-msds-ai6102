import os
import tensorflow as tf
import tensorflow_decision_forests as tfdf
from sklearn.metrics import classification_report
# from tensorflow import keras
# from tensorflow.keras import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class GradientBoostedTrees:
    def __init__(self, train_df:pd.DataFrame, valid_df:pd.DataFrame, test_df:pd.DataFrame, label:str, threshold:float=0.5):
        print("============ Instantiating GBT class ============")
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.label = label
        self.submission_id = test_df.PassengerId
        self.threshold = threshold
        
    def feature_selection(self, selected_features:list=None):
        """
        Function to update training, validation and test data with selected features
        """
        print("============ Pruning Features ============")
        self.selected_features = selected_features

        # update datasets
        self.train_df = self.train_df[selected_features]
        self.valid_df = self.valid_df[selected_features]
        selected_features.remove('Transported')
        self.test_df = self.test_df[selected_features]
        
    def create_tuner(self, num_trials:int=20):
        """ Function to create a Random Search tuner"""

        print("============ Creating RandomSearch Tuner ============")
        self._tuner = tfdf.tuner.RandomSearch(num_trials=num_trials, use_predefined_hps=True)

    def create_gbt_model(self):
        """ Function to instantiate GBT model"""

        print("============ Instantiating GBT model ============")
        ## convert pandas dataframe to tensorflow dataset
        self.train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(self.train_df, label=self.label)
        self.valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(self.valid_df,label=self.label)
        self.test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(self.test_df)

        # instantiate model with tuner
        self._model = tfdf.keras.GradientBoostedTreesModel(tuner=self._tuner)
        self._model.compile(metrics=["accuracy"]) # compile accuracy metrics

    def run_experiments (self):
        """Function to run experiments"""
        print("============ Running Experiment ============")
        self.history = self._model.fit(self.train_ds)
        return self.history

    def evaluate(self):
        """Function to evaluate model with validation set"""
        print("============ Evaluating ============")
        self.evaluation = self._model.evaluate(x=self.valid_ds,return_dict=True)
        y_pred = self._model.predict(self.valid_ds)
        y_true = np.reshape(self.valid_df[self.label].values, (-1,1))

        self.metrics = {}
        precision_metric = tf.keras.metrics.Precision()
        precision_metric.update_state(y_true,y_pred)
        self.metrics['Precision'] = precision_metric.result().numpy()

        f1_score = tf.keras.metrics.F1Score()
        f1_score.update_state(y_true,y_pred)
        self.metrics['f1_score'] = f1_score.result().numpy()[0]

        self.classification_report = classification_report(y_true=y_true, y_pred=(y_pred > self.threshold).astype(bool))

        return self.evaluation, self.metrics, self.classification_report

    def predict(self):
        """
        Function to make predictions
        
        Returns:
            predictions: 
            output [pandas dataframe]: submission output pd dataframe

        """
        print("============ Predicting ============")
        self.predictions = self._model.predict(self.test_ds)
        n_predictions = (self.predictions > self.threshold).astype(bool)
        self.output = pd.DataFrame({
            'PassengerId': self.submission_id,
            'Transported': n_predictions.squeeze()
            })
        
        print(self._model.summary())
        return self.predictions, self.output
    
    def plot_training_logs(self):
        """Visualise training logs of evaluated model"""

        self.training_logs = self._model.make_inspector().training_logs()

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot([log.num_trees for log in self.training_logs], [log.evaluation.accuracy for log in self.training_logs])
        plt.xlabel("Number of trees")
        plt.ylabel("Accuracy (out-of-bag)")

        plt.subplot(1, 2, 2)
        plt.plot([log.num_trees for log in self.training_logs], [log.evaluation.loss for log in self.training_logs])
        plt.xlabel("Number of trees")
        plt.ylabel("Logloss (out-of-bag)")

        plt.show()
        return self.training_logs
    
    def plot_tuning_logs (self):
        """Function to plot tuning logs to show optimal hyper parameters"""

        # Display the tuning logs.
        self.tuning_logs = self._model.make_inspector().tuning_logs()

        # Best hyper-parameters.
        print(self.tuning_logs[self.tuning_logs.best].iloc[0])

        # plots
        plt.figure(figsize=(10, 5))
        plt.plot(self.tuning_logs["score"], label="current trial")
        plt.plot(self.tuning_logs["score"].cummax(), label="best trial")
        plt.xlabel("Tuning step")
        plt.ylabel("Tuning score")
        plt.legend()
        plt.show()

        return self.tuning_logs
    
    def plot_variable_importances(self, key:str='INV_MEAN_MIN_DEPTH'):
        """
        Function to plot variable importance 

        Args:
            key [str]: key for variable importance. Available keys are: INV_MEAN_MIN_DEPTH, NUM_AS_ROOT, SUM_SCORE, NUM_NODES. Default is INV_MEAN_MIN_DEPTH
        """

        self.variable_importances = self._model.make_inspector().variable_importances()

        plt.figure(figsize=(12, 4))

        # Mean decrease in AUC of the class 1 vs the others.
        variable_importance_metric = key
        variable_importances = self.variable_importances[variable_importance_metric]

        # Extract the feature name and importance values.
        #
        # `variable_importances` is a list of <feature, importance> tuples.
        feature_names = [vi[0].name for vi in variable_importances]
        feature_importances = [vi[1] for vi in variable_importances]
        # The feature are ordered in decreasing importance value.
        feature_ranks = range(len(feature_names))

        bar = plt.barh(feature_ranks, feature_importances, label=[str(x) for x in feature_ranks])
        plt.yticks(feature_ranks, feature_names)
        plt.gca().invert_yaxis()

        # Label each bar with values
        for importance, patch in zip(feature_importances, bar.patches):
            plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{importance:.4f}", va="top")

        plt.xlabel(variable_importance_metric)
        plt.title("Mean decrease in AUC of the class 1 vs the others")
        plt.tight_layout()
        plt.show()

        return self.variable_importances
    

if __name__ == "__main__":
    GradientBoostedTrees