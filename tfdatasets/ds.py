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
        visualization.show_samples(ds_facade.train())

    # create tf records
    return ds_facade


class DatasetFacade:
    def __init__(self, ds_builder):
        self.ds_builder = ds_builder

    def train(self):
        return self.ds_builder.make_dataset('train')

    def validation(self):
        return self.ds_builder.make_dataset('validation')

    def validation(self):
        return self.ds_builder.make_dataset('eval')
