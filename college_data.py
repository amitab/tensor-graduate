import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

class Dataset(object):
    def __init__(self, file, categories):
        self.file = file
        f = open(self.file, 'r')
        self.columns = f.readline().strip().split(',')
        f.close()
        self.categories = categories

    def load_data(self, y_name, test_size=0.33):
        data = pd.read_csv(self.file, names=self.columns, header=0)
        data_x, data_y = data, data.pop(y_name)

        train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=test_size, random_state=42)

        return (train_x, train_y), (test_x, test_y)


    def train_input_fn(self, features, labels, batch_size):
        """An input function for training"""
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shuffle, repeat, and batch the examples.
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)

        # Return the dataset.
        return dataset


    def eval_input_fn(self, features, labels, batch_size):
        """An input function for evaluation or prediction"""
        features=dict(features)
        if labels is None:
            # No labels, use only features.
            inputs = features
        else:
            inputs = (features, labels)

        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices(inputs)

        # Batch the examples
        assert batch_size is not None, "batch_size must not be None"
        dataset = dataset.batch(batch_size)

        # Return the dataset.
        return dataset
