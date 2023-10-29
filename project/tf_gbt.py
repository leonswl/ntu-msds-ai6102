# tensorflow decision forests

import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd

label = 'Transported'

def tf_gbt (train_df:pd.DataFrame, valid_df:pd.DataFrame, test_df:pd.DataFrame):
    """
    Args:
        train_df, valid_df, test_df [pandas DataFrame]: pandas df of training, validation and test set
    
    Returns:
        rf []: tensorflow random forest model
        output [pandas DataFrame]: prediction output
    """
    ## load as tensorflow dataset
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label=label)
    valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_df,label=label)
    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df)

    ## Create a Random Search tuner with 50 trials and automatic hp configuration.
    tuner = tfdf.tuner.RandomSearch(num_trials=50, use_predefined_hps=True)

    model = tfdf.keras.GradientBoostedTreesModel(tuner=tuner,verbose=2)
    model.compile(metrics=["accuracy"]) # Optional, you can use this to include a list of eval metrics

    ## training the model
    history = model.fit(x=train_ds)
    print(f"Train Model Accuracy: {history.history['accuracy']}")

    ## Evaluate using validation set
    evaluation = model.evaluate(x=valid_ds,return_dict=True)
    evaluation_accuracy = evaluation['accuracy']
    print(f"Test accuracy with the TF-DF hyper-parameter tuner: {evaluation_accuracy:.4f}")

    # for name, value in evaluation.items():
    #     print(f"{name}: {value:.4f}")

    ## get predictions
    submission_id = test_df.PassengerId
    predictions = model.predict(test_ds)
    n_predictions = (predictions > 0.5).astype(bool)
    output = pd.DataFrame({
        'PassengerId': submission_id,
        'Transported': n_predictions.squeeze()
        })
    
    return model, output

def plot_training_logs (model):
    """
    Visualise training logs of evaluated model
    """
    import matplotlib.pyplot as plt

    logs = model.make_inspector().training_logs()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy (out-of-bag)")

    plt.subplot(1, 2, 2)
    plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Logloss (out-of-bag)")

    plt.show()

def plot_tuning_logs (model):
    """
    function to plot tuning logs to show optimal hyper parameters
    """
    import matplotlib.pyplot as plt

    # Display the tuning logs.
    tuning_logs = model.make_inspector().tuning_logs()

    # Best hyper-parameters.
    print(tuning_logs[tuning_logs.best].iloc[0])

    # plots
    plt.figure(figsize=(10, 5))
    plt.plot(tuning_logs["score"], label="current trial")
    plt.plot(tuning_logs["score"].cummax(), label="best trial")
    plt.xlabel("Tuning step")
    plt.ylabel("Tuning score")
    plt.legend()
    plt.show()

def main():
    """ Main function"""
    ## load train and validation dataset
    train_df = pd.read_csv("data/train_ds_pd.csv")
    valid_df = pd.read_csv("data/valid_ds_pd.csv")
    test_df = pd.read_csv("data/test_ds_pd.csv")

    # tensorflow random forest
    model, output = tf_gbt(train_df=train_df, valid_df=valid_df, test_df=test_df)

    plot_training_logs(model)

    plot_tuning_logs(model)

    output.to_csv("submissions/tf_gbt.csv",index=False)

if __name__ == "__main__":
    main()

