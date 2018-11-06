import logging
import os
import pickle
import sys
import tensorflow as tf
from tfdatasets.pipelines.helpers.downloader import download_and_extract
from tfdatasets.pipelines.helpers.logger_handler import set_basic_config

logger = logging.getLogger(__name__)
set_basic_config(logger)


class ModelLoader:
    def __init__(self, data_dir, logging_level,
                 dataset_download_url, dataset_local_folder, dataset_filename=None):
        self.data_dir = data_dir
        self.logging_level = logging_level
        logger.setLevel(self.logging_level)
        self.dataset_filename = dataset_filename or dataset_download_url[dataset_download_url.rfind('/') + 1:]
        self.dataset_download_url = dataset_download_url
        self.dataset_local_folder = dataset_local_folder

    def get_file_names(self):
        raise NotImplemented()

    def load(self):
        logger.info(f'Download from {self.dataset_download_url} and extract to {self.data_dir}.')

        download_and_extract(self.data_dir,
                             download_url=self.dataset_download_url,
                             filename=self.dataset_filename)

        self.convert_to_tfrecords()

    def convert_to_tfrecords(self):
        file_names = self.get_file_names()
        for mode, files in file_names.items():
            input_files = [os.path.join(self.data_dir, *f) for f in files]
            output_file = os.path.join(self.data_dir, mode + '.tfrecords')
            try:
                os.remove(output_file)
            except OSError:
                pass
            # Convert to tf.train.Example and write the to TFRecords.
            self.convert_to_tfrecord(input_files, output_file)
        logger.info('Done!')

    def convert_to_tfrecord(self, input_files, output_file):
        """
        Converts a file to TFRecords.
        """
        logger.info('Generating %s' % output_file)
        with tf.python_io.TFRecordWriter(output_file) as record_writer:
            for input_file in input_files:
                self.process_input_file(record_writer, input_file)

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _read_pickle_from_file(self, filename):
        with tf.gfile.Open(filename, 'rb') as f:
            if sys.version_info >= (3, 0):
                data_dict = pickle.load(f, encoding='bytes')
            else:
                data_dict = pickle.load(f)
        return data_dict
