import os
from tfdatasets.visualization import visualization
from tfdatasets.pipelines import get_loader_by_name, get_processor_by_name


def get_dataset(name, path=None, show_samples=False, logging_level='INFO'):
    # load dataset
    path = os.path.join(path, name + '-tf-data')
    loader = get_loader_by_name(name)
    loader.main(data_dir=path, logging_level=logging_level)

    processor = get_processor_by_name(name)

    ds_facade = DatasetFacade(processor.TFDataSetBuilder(path))

    # show samples
    if show_samples:
        visualization.show_samples(ds_facade.train(
            batch=16, shuffle=None, repeat=None)())

    # create tf records
    return ds_facade


class DatasetFacade:
    def __init__(self, ds_builder):
        self.ds_builder = ds_builder

    def get_all_feature_columns(self):
        return self.ds_builder.get_all_feature_columns()

    def num_of_classes(self):
        # TODO:
        return None

    def train(self, batch=16, shuffle=1000, repeat=True):
        def train_input_fn():
            ds = self.ds_builder.make_dataset('train')
            if batch is not None:
                ds = ds.batch(batch)
            if repeat:
                ds = ds.repeat()
            if shuffle is not None:
                ds = ds.shuffle(shuffle)
            return ds
        return train_input_fn

    def validation(self):
        return self.ds_builder.make_dataset('validation')

    def validation(self, batch=16):
        def train_input_fn():
            ds = self.ds_builder.make_dataset('eval')
            if batch is not None:
                ds = ds.batch(batch)
            return ds
        return train_input_fn
