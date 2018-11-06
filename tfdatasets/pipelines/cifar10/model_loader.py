# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Read CIFAR-10 data from pickled numpy arrays and writes TFRecords.

Generates tf.train.Example protos and writes them to TFRecord files from the
python version of the CIFAR-10 dataset downloaded from
https://www.cs.toronto.edu/~kriz/cifar.html.
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import tensorflow as tf
from tfdatasets.pipelines.helpers import model_loader
from tfdatasets.pipelines.helpers.logger_handler import set_basic_config

logger = logging.getLogger(__name__)
set_basic_config(logger)

DATASET_FILENAME = 'cifar-10-python.tar.gz'
DATASET_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/' + DATASET_FILENAME
DATASET_LOCAL_FOLDER = 'cifar-10-batches-py'


def load_model(data_dir, logging_level='INFO'):
    logger.setLevel(logging_level)

    CIFAR10ModelLoader(data_dir, logging_level,
                       dataset_download_url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                       dataset_local_folder='cifar-10-batches-py').load()


class CIFAR10ModelLoader(model_loader.ModelLoader):
    def get_file_names(self):
        """Returns the file names expected to exist in the input_dir."""
        file_names = {}
        file_names['train'] = [['cifar-10-batches-py', f'data_batch_{i}'] for i in range(1, 5)]
        file_names['validation'] = [['cifar-10-batches-py', 'data_batch_5']]
        file_names['eval'] = [['cifar-10-batches-py', 'test_batch']]
        return file_names

    def process_input_file(self, record_writer, input_file):
        data_dict = self._read_pickle_from_file(input_file)
        data = data_dict[b'data']
        labels = data_dict[b'labels']
        num_entries_in_batch = len(labels)
        for i in range(num_entries_in_batch):
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image': self._bytes_feature(data[i].tobytes()),
                    'label': self._int64_feature(labels[i])
                }))
            record_writer.write(example.SerializeToString())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        type=str,
        default='',
        help='Directory to download and extract CIFAR-10 to.')
    parser.add_argument(
        '--log-level',
        type=str,
        default='',
        help='Level of logging, could be (DEBUG, INFO, WARNING and ERROR)')

    args = parser.parse_args()
    load_model(args.data_dir, args.log_level)
