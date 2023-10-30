## Spaceship Titanic

Check out the [Spaceship Titanic Competition](https://www.kaggle.com/competitions/spaceship-titanic) for more information.

### Project Directory
- [data](data): all input data files are stored here
- [submissions](submissions): output submission data files are stored here

### Models
- [Tensorflow RF](tf_rf.py): Tensorflow RandomForest model in default state for baseline reference.
- [Tensorflow GBT](tf_gbt.py): Tensorflow GradientBoosterTree model with auto hyperparameter tuning and feature selection. 
- [Tensorflow GBT with Target Encoding](tf_gbt_te.py): Tensorflow GradientBoosterTree model with auto hyperparameter tuning and Targeting Encoding.

#### Tensorflow RandomForest (RF)

This model was replicated from the [starter notebook](https://www.kaggle.com/code/gusthema/spaceship-titanic-with-tfdf/notebook) to derive a quick baseline for further experiments.

Code from the notebook were refactored and optimised for quick experimental executions.

#### Tensorflow Gradient Booster Tree (GBT)

This model was experiment using a variety of tricks:
- manual/auto hyperparameter tuning
- feature selection by analysing variable importance across various keys

#### Tensorflow GBT with Target Encoding

This model is a work in progress, with the code from [keras documentation](https://keras.io/examples/structured_data/classification_with_tfdf/#implement-a-feature-encoding-with-target-encoding). 

