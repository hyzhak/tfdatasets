from tfdatasets.pipelines.iris.model_loader import load_model
from tfdatasets.pipelines.iris.processor import TFDataSetBuilder

num_of_classes = 3

__all__ = [
    load_model,
    num_of_classes,
    TFDataSetBuilder,
]
