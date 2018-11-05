from tfdatasets.pipelines.cifar10.model_loader import main as load_model
from tfdatasets.pipelines.cifar10.processor import TFDataSetBuilder

num_of_classes = 10

__all__ = [
    load_model,
    num_of_classes,
    TFDataSetBuilder,
]
