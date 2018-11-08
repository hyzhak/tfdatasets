import os
from tfdatasets.pipelines import get_profile


def get_dataset(name, data_dir=None, show_samples=False, logging_level='INFO'):
    # load dataset
    data_dir = os.path.join(data_dir, name + '-tf-data')

    ds_profile = get_profile(name)
    ds_profile.load_model(data_dir=data_dir, logging_level=logging_level)
    ds_facade = DatasetFacade(ds_profile, ds_profile.TFDataSetBuilder(data_dir))

    # show samples
    if show_samples:
        ds_facade.show_samples()

    # create tf records
    return ds_facade


class DatasetFacade:
    def __init__(self, ds_profile, ds_builder):
        self.ds_profile = ds_profile
        self.ds_builder = ds_builder

    def get_all_feature_columns(self):
        return self.ds_builder.get_all_feature_columns()

    def num_of_classes(self):
        return self.ds_profile.num_of_classes

    def show_samples(self, subset='train', limit=32):
        return self.ds_profile.show_samples(self.ds_builder.make_dataset(subset),
                                            limit)

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

    def validation(self, batch=None):
        def validation_input_fn():
            ds = self.ds_builder.make_dataset('validation')
            if batch is not None:
                ds = ds.batch(batch)
            return ds

        return validation_input_fn

    def eval(self, batch=16):
        def eval_input_fn():
            ds = self.ds_builder.make_dataset('eval')
            if batch is not None:
                ds = ds.batch(batch)
            return ds

        return eval_input_fn
