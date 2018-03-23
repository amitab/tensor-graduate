#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a DNNClassifier for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

from college_data import Dataset


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')
parser.add_argument('--file', required=True, type=str, help='data file')
parser.add_argument('--light', default=False, action='store_true')
parser.add_argument('--papers', default=False, action='store_true')

def main(argv):
    args = parser.parse_args(argv[1:])
    dataset = Dataset(args.file, ['REJECT', 'ADMIT'])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = dataset.load_data('status')

    # Feature columns describe how to use the input.
    deep_columns, linear_columns = dataset.get_features(args.light, args.papers)

    classifier = tf.estimator.DNNLinearCombinedClassifier(
        linear_feature_columns=linear_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[20, 40, 20],
        n_classes=2)

    # Train the Model.
    classifier.train(
        input_fn=lambda:dataset.train_input_fn(train_x, train_y, args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:dataset.eval_input_fn(test_x, test_y, args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    predict_x = {
        'gre': [324],
        'english': [7.5],
        'undergrad': [8.66],
        'work': [28],
        'season': ['Fall']
    }
    if not args.light:
        predict_x['quant'] = [162]
        predict_x['verbal'] = [162]
        if args.papers:
            predict_x['papers'] = ['None']

    predictions = classifier.predict(
        input_fn=lambda:dataset.eval_input_fn(predict_x,
                                                   labels=None,
                                                   batch_size=args.batch_size))

    for pred_dict in predictions:
        template = ('\nPrediction is "{}" ({:.1f}%)')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(dataset.categories[class_id], 100 * probability))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
