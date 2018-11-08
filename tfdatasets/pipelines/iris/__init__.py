from tfdatasets.pipelines.iris.model_loader import load_model
from tfdatasets.pipelines.iris.processor import TFDataSetBuilder
from tfdatasets.visualization import table_visualization

num_of_classes = 3


def show_samples(ds):
    return table_visualization.show_samples(ds)


__all__ = [
    load_model,
    num_of_classes,
    TFDataSetBuilder,
]
