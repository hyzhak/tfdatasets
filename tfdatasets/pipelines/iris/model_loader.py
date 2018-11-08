import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tfdatasets.pipelines.helpers import model_loader
from tfdatasets.pipelines.helpers.logger_handler import set_basic_config

logger = logging.getLogger(__name__)
set_basic_config(logger)

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


def load_model(data_dir, logging_level='INFO'):
    logger.setLevel(logging_level)
    IrisModelLoader(data_dir, logging_level,
                    dataset_download_url='http://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data',
                    dataset_local_folder='iris-batches-py').load()


def label_name_to_id(name):
    """
    get id of label by its name
    :param name:
    :return:
    """
    return SPECIES.index(name)


class IrisModelLoader(model_loader.ModelLoader):
    def convert_to_tfrecords(self):
        """
        overwrite default behaviour of model_loader.ModelLoader
        because we have here mush simpler case
        source batch file -> split to 3 train, validate, eval

        TODO: once we got more similar case
        1) add support of cross validation
        2) common solution for any csv dataset

        :return:
        """
        # read dataset file
        input_file = os.path.join(self.data_dir, self.dataset_filename)
        csv = pd.read_csv(input_file, names=CSV_COLUMN_NAMES)

        # split to train, test, validate (datasets) subsets

        # 150 -> 120 + 15 + 15 (parts: 8/10, 1/10, 1/10)
        #
        # 150 -> 135 + 15 (parts: 9/10 , 1/10)
        train_and_validation, test = train_test_split(csv, test_size=1 / 10)

        # TODO: we can use cross validation here
        # from sklearn.cross_validation import train_test_split
        # 135 -> 120 + 15 (parts: 8/9 + 1/9)
        train, validation = train_test_split(train_and_validation, test_size=1 / 9)

        subsets = {
            'train': train,
            'validation': validation,
            'eval': test,
        }

        # store to tfrecords
        for mode, subset in subsets.items():
            output_file = os.path.join(self.data_dir, mode + '.tfrecords')
            try:
                os.remove(output_file)
            except OSError:
                pass
            logger.info(f'Generating {output_file}')
            with tf.python_io.TFRecordWriter(output_file) as record_writer:
                self.process_input_values(record_writer,
                                          csv.columns.values,
                                          subset.values)

    def process_input_values(self, record_writer, columns, values):
        for row in values:
            input_features, label = row[:-1], row[-1]

            # TODO: we got 5 times bigger tfrecords files
            # so maybe we should skip feature names?

            features = {}
            # all input features are float
            for col_name, feature_value in zip(columns[:-1], input_features):
                features[col_name] = self.build_float_feature(feature_value)

            # but label has int type
            features[columns[-1]] = self.build_int64_feature(label_name_to_id(label))

            example = tf.train.Example(features=tf.train.Features(feature=features))
            record_writer.write(example.SerializeToString())
