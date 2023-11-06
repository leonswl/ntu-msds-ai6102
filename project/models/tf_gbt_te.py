# Tensorflow GradientBoosterTree with Target Encoding 

import math
import urllib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_decision_forests as tfdf


TARGET_COLUMN_NAME = 'Transported'

NUMERIC_FEATURE_NAMES = ['CryoSleep','Age','VIP','RoomService','Cabin_num','FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

CATEGORICAL_FEATURE_NAMES = ['HomePlanet','Destination', 'Side', 'Deck']

def run_experiment(model, train_data, valid_data, num_epochs=1, batch_size=None):

    ## load as tensorflow dataset
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_data, label=TARGET_COLUMN_NAME)
    valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_data,label=TARGET_COLUMN_NAME)

    model.fit(train_ds, epochs=num_epochs, batch_size=batch_size)
    _, accuracy = model.evaluate(valid_ds, verbose=0)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

class BinaryTargetEncoding(layers.Layer):
    def __init__(self, vocabulary_size=None, correction=1.0, **kwargs):
        super().__init__(**kwargs)
        self.vocabulary_size = vocabulary_size
        self.correction = correction

    def adapt(self, data):
        # data is expected to be an integer numpy array to a Tensor shape [num_exmples, 2].
        # This contains feature values for a given feature in the dataset, and target values.

        # Convert the data to a tensor.
        data = tf.convert_to_tensor(data)
        # Separate the feature values and target values
        feature_values = tf.cast(data[:, 0], tf.dtypes.int32)
        target_values = tf.cast(data[:, 1], tf.dtypes.bool)

        # Compute the vocabulary_size of not specified.
        if self.vocabulary_size is None:
            self.vocabulary_size = tf.unique(feature_values).y.shape[0]

        # Filter the data where the target label is positive.
        positive_indices = tf.where(condition=target_values)
        postive_feature_values = tf.gather_nd(
            params=feature_values, indices=positive_indices
        )
        # Compute how many times each feature value occurred with a positive target label.
        positive_frequency = tf.math.unsorted_segment_sum(
            data=tf.ones(
                shape=(postive_feature_values.shape[0], 1), dtype=tf.dtypes.float64
            ),
            segment_ids=postive_feature_values,
            num_segments=self.vocabulary_size,
        )

        # Filter the data where the target label is negative.
        negative_indices = tf.where(condition=tf.math.logical_not(target_values))
        negative_feature_values = tf.gather_nd(
            params=feature_values, indices=negative_indices
        )
        # Compute how many times each feature value occurred with a negative target label.
        negative_frequency = tf.math.unsorted_segment_sum(
            data=tf.ones(
                shape=(negative_feature_values.shape[0], 1), dtype=tf.dtypes.float64
            ),
            segment_ids=negative_feature_values,
            num_segments=self.vocabulary_size,
        )
        # Compute positive probability for the input feature values.
        positive_probability = positive_frequency / (
            positive_frequency + negative_frequency + self.correction
        )
        # Concatenate the computed statistics for traget_encoding.
        target_encoding_statistics = tf.cast(
            tf.concat(
                [positive_frequency, negative_frequency, positive_probability], axis=1
            ),
            dtype=tf.dtypes.float32,
        )
        self.target_encoding_statistics = tf.constant(target_encoding_statistics)

    def call(self, inputs):
        # inputs is expected to be an integer numpy array to a Tensor shape [num_exmples, 1].
        # This includes the feature values for a given feature in the dataset.

        # Raise an error if the target encoding statistics are not computed.
        if self.target_encoding_statistics == None:
            raise ValueError(
                f"You need to call the adapt method to compute target encoding statistics."
            )

        # Convert the inputs to a tensor.
        inputs = tf.convert_to_tensor(inputs)
        # Cast the inputs int64 a tensor.
        inputs = tf.cast(inputs, tf.dtypes.int64)
        # Lookup target encoding statistics for the input feature values.
        target_encoding_statistics = tf.cast(
            tf.gather_nd(self.target_encoding_statistics, inputs),
            dtype=tf.dtypes.float32,
        )
        return target_encoding_statistics
    
def create_model_inputs():
    inputs = {}

    for feature_name in NUMERIC_FEATURE_NAMES:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(), dtype=tf.float32
        )

    for feature_name in CATEGORICAL_FEATURE_NAMES:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(), dtype=tf.string
        )

    return inputs
    
def create_target_encoder(train_data):
    inputs = create_model_inputs()
    target_values = train_data[[TARGET_COLUMN_NAME]].to_numpy()
    encoded_features = []
    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURE_NAMES:
            print(feature_name)
            # Get the vocabulary of the categorical feature.
            vocabulary = sorted(
                [str(value) for value in list(train_data[feature_name].unique())]
            )
            # Create a lookup to convert string values to an integer indices.
            # Since we are not using a mask token nor expecting any out of vocabulary
            # (oov) token, we set mask_token to None and  num_oov_indices to 0.
            lookup = layers.StringLookup(
                vocabulary=vocabulary, mask_token=None, num_oov_indices=1
            )
            # Convert the string input values into integer indices.
            value_indices = lookup(inputs[feature_name])
            # Prepare the data to adapt the target encoding.
            print("### Adapting target encoding for:", feature_name)
            feature_values = train_data[[feature_name]].to_numpy().astype(str)
            feature_value_indices = lookup(feature_values)
            data = tf.concat([feature_value_indices, target_values], axis=1)
            feature_encoder = BinaryTargetEncoding()
            feature_encoder.adapt(data)
            # Convert the feature value indices to target encoding representations.
            encoded_feature = feature_encoder(tf.expand_dims(value_indices, -1))
        else:
            # Expand the dimensions of the numerical input feature and use it as-is.
            encoded_feature = tf.expand_dims(inputs[feature_name], -1)
        # Add the encoded feature to the list.
        encoded_features.append(encoded_feature)
    # Concatenate all the encoded features.
    encoded_features = tf.concat(encoded_features, axis=1)
    # Create and return a Keras model with encoded features as outputs.
    return keras.Model(inputs=inputs, outputs=encoded_features)


def create_gbt_with_preprocessor(preprocessor):

    ## Create a Random Search tuner with 50 trials and automatic hp configuration.
    tuner = tfdf.tuner.RandomSearch(num_trials=50, use_predefined_hps=True)

    gbt_model = tfdf.keras.GradientBoostedTreesModel(
        preprocessing=preprocessor,
        tuner=tuner
    )

    gbt_model.compile(metrics=["accuracy"])

    return gbt_model

def main():

    # load data
    train_df = pd.read_csv("data/train_ds_pd.csv")
    valid_df = pd.read_csv("data/valid_ds_pd.csv")
    # test_df = pd.read_csv("data/test_ds_pd.csv")

    # test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df)


    gbt_model = create_gbt_with_preprocessor(create_target_encoder(train_df))
    run_experiment(gbt_model, train_df, valid_df)


if __name__ == "__main__":
    main()