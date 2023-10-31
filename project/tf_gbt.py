# tensorflow decision forests
import os
import tensorflow_decision_forests as tfdf
import pandas as pd
import matplotlib.pyplot as plt

class GradientBoostedTrees:
    def __init__(self, train_df:pd.DataFrame, valid_df:pd.DataFrame, test_df:pd.DataFrame, label:str):
        print("============ Instantiating GBT class ============")
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.label = label
        self.submission_id = test_df.PassengerId
        

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

        return self.evaluation

    def predict(self):
        """
        Function to make predictions
        
        Returns:
            predictions: 
            output [pandas dataframe]: submission output pd dataframe

        """
        print("============ Predicting ============")
        self.predictions = self._model.predict(self.test_ds)
        n_predictions = (self.predictions > 0.5).astype(bool)
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

def main():
    ## load train and validation dataset
    train_df = pd.read_csv("data/train_ds_pd.csv")
    valid_df = pd.read_csv("data/valid_ds_pd.csv")
    test_df = pd.read_csv("data/test_ds_pd.csv")

    select_features = ['CryoSleep','Age','RoomService','Cabin_num','FoodCourt', 'ShoppingMall', 'Spa', 'HomePlanet', 'Side', 'Deck', 'Transported', 'VRDeck','Destination']

    label = 'Transported'

    gbt = GradientBoostedTrees(train_df=train_df, valid_df=valid_df, test_df=test_df,label=label)
    gbt.feature_selection(selected_features=select_features)
    gbt.create_tuner(num_trials=50)
    gbt.create_gbt_model()

    # run experiment
    gbt_model_history = gbt.run_experiments()
    print(f"Train Model Accuracy: {gbt_model_history.history['accuracy']}")

    # evaluate
    gbt_model_evaluation = gbt.evaluate()
    evaluation_accuracy = gbt_model_evaluation['accuracy']
    print(f"Test accuracy with the TF-DF hyper-parameter tuner: {evaluation_accuracy:.4f}")

    # predict
    gbt_model_predictions, gbt_model_output = gbt.predict()

    # training logs
    gbt_model_training_logs = gbt.plot_training_logs()

    # tuning logs
    gbt_model_tuning_logs = gbt.plot_tuning_logs()

    # variable importance
    gbt_model_variable_importances = gbt.plot_variable_importances()

    os.makedirs('submissions', exist_ok=True) 
    gbt_model_output.to_csv("submissions/tf_gbt.csv",index=False)

if __name__ == "__main__":
    main()

