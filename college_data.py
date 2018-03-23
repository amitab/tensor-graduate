import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

deep_features = {
    'gre': tf.feature_column.numeric_column(key='gre'),
    'work': tf.feature_column.numeric_column(key='work'),
    'quant': tf.feature_column.numeric_column(key='quant'),
    'verbal': tf.feature_column.numeric_column(key='verbal'),
    'english': tf.feature_column.numeric_column(key='english'),
    'undergrad': tf.feature_column.numeric_column(key='undergrad'),
    'season': tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            key="season",
            vocabulary_list=["Fall", "Spring"])),
    'papers': tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            key="papers",
            vocabulary_list=["international", "national", "local", "none"]))
}

linear_features = {
    'quant_buckets': tf.feature_column.bucketized_column(
        deep_features['quant'],
        boundaries=[135, 140, 145, 150, 155, 160, 165]),
    'verbal_buckets': tf.feature_column.bucketized_column(
        deep_features['verbal'],
        boundaries=[135, 140, 145, 150, 155, 160, 165]),
    'gre_buckets': tf.feature_column.bucketized_column(
        deep_features['gre'],
        boundaries=[270, 280, 290, 300, 310, 320, 330]),
    'work_buckets': tf.feature_column.bucketized_column(
        deep_features['work'],
        boundaries=[10, 20, 30]),
    'ug_buckets': tf.feature_column.bucketized_column(
        deep_features['undergrad'],
        boundaries=[5, 7, 9]),
    'english_buckets': tf.feature_column.bucketized_column(
        deep_features['english'],
        boundaries=[4, 6, 8])
}

class Dataset(object):
    def __init__(self, file, categories):
        self.file = file
        self.categories = categories

        f = open(self.file, 'r')
        self.columns = f.readline().strip().split(',')
        f.close()

        data = pd.read_csv(self.file, names=self.columns, header=0)
        q_buckets = tf.feature_column.bucketized_column(
            tf.feature_column.numeric_column('quant'),
            sorted(data['quant'].unique().tolist()))
        v_buckets = tf.feature_column.bucketized_column(
            tf.feature_column.numeric_column('verbal'),
            sorted(data['verbal'].unique().tolist()))
        linear_features['cross_q_v'] = tf.feature_column.crossed_column(
            [q_buckets, v_buckets], 5000)
    
    def get_features(self, light=False, papers=False):
        deep_columns = [
            deep_features['english'],
            deep_features['undergrad'],
            deep_features['work'],
            deep_features['gre'],
            deep_features['season']]
        linear_columns = [
            linear_features['english_buckets'],
            linear_features['ug_buckets'],
            linear_features['work_buckets'],
            linear_features['gre_buckets'],
            linear_features['cross_q_v']]

        if not light:
            deep_columns.extend([
                deep_features['quant'],
                deep_features['verbal']])
            linear_columns.extend([
                linear_features['quant_buckets'],
                linear_features['verbal_buckets']])
            if papers:
                deep_columns.append(deep_features['papers'])

        return deep_columns, linear_columns

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
