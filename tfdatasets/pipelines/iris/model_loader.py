import logging
import pandas as pd
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
    def get_file_names(self):
        # TODO: split to train, test, validate (datasets)
        # 100, 25, 25
        return {
            'train': [['bezdekIris.data']],
            'validation': [['bezdekIris.data']],
            'eval': [['bezdekIris.data']],
        }

    def process_input_file(self, record_writer, input_file):
        csv = pd.read_csv(input_file, names=CSV_COLUMN_NAMES)

        # TODO: should split data csv to train, test and validate sub datasets
        # 1) shuffle (once for all sub-datasets)
        # 2) split to parts

        for row in csv.values:
            features, label = row[:-1], row[-1]
            example = tf.train.Example()
            label_id = SPECIES.index(label)
            example.features.feature['features'].float_list.value.extend(features)
            example.features.feature['label'].int64_list.value.append(label_id)
            record_writer.write(example.SerializeToString())
