
import os
import pandas as pd
from models.tf_gbt import GradientBoostedTrees

def load_datasets(data_path):
    train_df = pd.read_csv(os.path.join(data_path, "train_ds_pd.csv"))
    valid_df = pd.read_csv(os.path.join(data_path, "valid_ds_pd.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test_ds_pd.csv"))

    return train_df, valid_df, test_df

# sourcery skip: avoid-builtin-shadow
data_path = "data"
output_path = "submissions"

train_df, valid_df, test_df = load_datasets(data_path)

select_features = ['CryoSleep','Age','RoomService','Cabin_num','FoodCourt', 'ShoppingMall', 'Spa', 'HomePlanet', 'Side', 'Deck', 'Transported', 'VRDeck','Destination']

label = 'Transported'

try:
    input = int(input("""
                  Select model for execution: 
                  - 1: TensorFlow GBT
                  """))
    if (input >= 1) and (input <=3):
        if input == 1:
            print('Executing Tensorflow Gradient Boosting Trees')
            model = GradientBoostedTrees(train_df=train_df, valid_df=valid_df, test_df=test_df,label=label)
            model.feature_selection(selected_features=select_features)
            model.create_tuner(num_trials=50)
            model.create_gbt_model()

            # run experiment
            model_history = model.run_experiments()
            print(f"Train Model Accuracy: {model_history.history['accuracy']}")

            # evaluate
            model_evaluation, model_metrics, model_classification_report = model.evaluate()
            evaluation_accuracy = model_evaluation['accuracy']
            print(f"Test accuracy with the TF-DF hyper-parameter tuner: {evaluation_accuracy:.4f}")

            # predict
            model_predictions, model_output = model.predict()

        elif input == 2:
            print('Executing Tensorflow Random Forest')
except Exception as e:
    print(f"{e}; Please input a valid request. Only integers are accepted.")
    exit
finally:
    os.makedirs('submissions', exist_ok=True)
    model_output.to_csv(os.path.join(output_path, "tf_gbt.csv"),index=False)

