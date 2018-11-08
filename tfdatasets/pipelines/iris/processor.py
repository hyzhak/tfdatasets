import os
import tensorflow as tf

from tfdatasets.pipelines.iris.model_loader import CSV_COLUMN_NAMES


class TFDataSetBuilder:
    def __init__(self, data_dir, use_distortion=True):
        self.data_dir = data_dir
        self.use_distortion = use_distortion

    def get_all_feature_columns(self):
        # get single sample from data set
        # and extract its fields
        one_sample_graph = self.make_dataset('train').batch(1).make_one_shot_iterator().get_next()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            one_sample = sess.run(one_sample_graph)
            # Feature columns describe how to use the input.

            # TODO: find the better way to represent features
            # dictionary like features
            my_feature_columns = []
            for key in one_sample[0].keys():
                my_feature_columns.append(tf.feature_column.numeric_column(key=key))

            # or
            # all features in a single column but with big shape
            # my_feature_columns = tf.feature_column.numeric_column(key='features',
            #                                                       shape=len(one_sample[0]))

        return my_feature_columns

    def get_filenames(self, subset):
        """
        put to the common class

        :param subset:
        :return:
        """
        if subset in ['train', 'validation', 'eval']:
            return [os.path.join(self.data_dir, subset + '.tfrecords')]
        else:
            raise ValueError('Invalid data subset "%s"' % subset)

    def parser(self, serialized_example):
        """
        Parses a single tf.Example into features and label tensors.
        """

        features_spec = {}
        # regular features all of them are float
        for col_name in CSV_COLUMN_NAMES[:-1]:
            features_spec[col_name] = tf.FixedLenFeature([], tf.float32)

        # label has different type
        features_spec[CSV_COLUMN_NAMES[-1]] = tf.FixedLenFeature([], tf.int64)

        features = tf.parse_single_example(
            serialized_example,
            features=features_spec)

        # input_features = [tf.cast(features[col_name], tf.float32) for col_name in CSV_COLUMN_NAMES[:-1]]

        input_features = {}
        for col_name in CSV_COLUMN_NAMES[:-1]:
            input_features[col_name] = tf.cast(features[col_name], tf.float32)

        label = tf.cast(features[CSV_COLUMN_NAMES[-1]], tf.int32)

        # TODO: Custom preprocessing

        return input_features, label

    def make_dataset(self, subset):
        file_names = self.get_filenames(subset)
        dataset = tf.data.TFRecordDataset(file_names)
        dataset = dataset.map(self.parser)
        return dataset
