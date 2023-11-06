# tensorflow decision forests

import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd



def tf_rf (train_df:pd.DataFrame, valid_df:pd.DataFrame, test_df:pd.DataFrame):
    """
    Args:
        train_df, valid_df, test_df [Pandas DataFrame]: pandas df of training, validation and test set
    
    Returns:
        rf []: tensorflow random forest model
        output []:
    """

    label = 'Transported'
    ## load as tensorflow dataset
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label=label)
    valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_df,label=label)
    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df)

    rf = tfdf.keras.RandomForestModel(verbose=2)
    rf.compile(metrics=["accuracy"]) # Optional, you can use this to include a list of eval metrics

    ## training the model
    history = rf.fit(x=train_ds)
    print(f"Train Model Accuracy: {history.history['accuracy']}")

    ## visualise model
    tfdf.model_plotter.plot_model_in_colab(rf, tree_idx=0, max_depth=3)

    ## Evaluate using validation set
    evaluation = rf.evaluate(x=valid_ds,return_dict=True)

    for name, value in evaluation.items():
        print(f"{name}: {value:.4f}")

    ## get predictions
    submission_id = test_df.PassengerId
    predictions = rf.predict(test_ds)
    n_predictions = (predictions > 0.5).astype(bool)
    output = pd.DataFrame({
        'PassengerId': submission_id,
        'Transported': n_predictions.squeeze()
        })
    
    return rf, output

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

def main():

    ## load train and validation dataset
    train_df = pd.read_csv("data/train_ds_pd.csv")
    valid_df = pd.read_csv("data/valid_ds_pd.csv")
    test_df = pd.read_csv("data/test_ds_pd.csv")

    # tensorflow random forest
    model, output = tf_rf(train_df=train_df, valid_df=valid_df, test_df=test_df)

    plot_training_logs(model)

    output.to_csv("submissions/tf_rf.csv",index=False)

if __name__ == "__main__":
    main()

