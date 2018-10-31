from tfdatasets.pipelines import get_pipeline


def get_dataset(name, path=None, show_samples=False, logging_level='INFO'):
    # TODO: load date

    pipeline = get_pipeline(name)
    pipeline.main(data_dir=path, logging_level=logging_level)

    # TODO: create tf records

    # TODO: show samples
    if show_samples:
        pass

    return DatasetInterface()


class DatasetInterface:
    def __init__(self):
        pass

    def train(self):
        pass

    def validation(self):
        pass

    def validation(self):
        pass
