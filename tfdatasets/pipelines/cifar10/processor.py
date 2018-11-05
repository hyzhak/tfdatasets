import os
import tensorflow as tf

HEIGHT = 32
WIDTH = 32
DEPTH = 3


class TFDataSetBuilder:
    def __init__(self, data_dir, use_distortion=True):
        self.data_dir = data_dir
        self.use_distortion = use_distortion

    def get_all_feature_columns(self):
        return [tf.feature_column.numeric_column('image', shape=[DEPTH, HEIGHT, WIDTH])]

    def make_dataset(self, subset):
        file_names = self._get_filenames(subset)
        dataset = tf.data.TFRecordDataset(file_names)
        dataset = dataset.map(self._parser)
        return dataset

    def _get_filenames(self, subset):
        if subset in ['train', 'validation', 'eval']:
            return [os.path.join(self.data_dir, subset + '.tfrecords')]
        else:
            raise ValueError('Invalid data subset "%s"' % subset)

    def _parser(self, serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        # Dimensions of the images in the CIFAR-10 dataset.
        # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
        # input format.
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })
        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([DEPTH * HEIGHT * WIDTH])

        # Reshape from [depth * height * width] to [depth, height, width].
        image = tf.cast(
            tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
            tf.float32)
        label = tf.cast(features['label'], tf.int32)

        # TODO: may left for next dataset.map
        # Custom preprocessing.
        # image = self.preprocess(image)

        return {'image': image}, label
#
#
# def make_dataset(dataset_path, subset):
#     return TFDataSetBuilder(dataset_path).make_dataset(subset)
