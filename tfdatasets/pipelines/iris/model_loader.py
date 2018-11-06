import logging
import os
import numpy as np
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
        train_and_validate, test = train_test_split(csv, test_size=1 / 10)

        # TODO: we can use cross validation here
        # from sklearn.cross_validation import train_test_split
        # 135 -> 120 + 15 (parts: 8/9 + 1/9)
        train, validate = train_test_split(train_and_validate, test_size=1 / 9)

        subsets = {
            'train': train,
            'eval': validate,
            'test': test,
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
                self.process_input_values(record_writer, subset.values)

    def process_input_values(self, record_writer, values):
        for row in values:
            features, label = row[:-1], row[-1]
            example = tf.train.Example()
            label_id = SPECIES.index(label)
            # TODO: choose better datatype here:
            # maybe here I could find more
            # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py
            example.features.feature['features'].float_list.value.extend(features)
            # TODO: and there:
            tf.train.Feature()
            example.features.feature['label'].int64_list.value.append(label_id)
            record_writer.write(example.SerializeToString())
