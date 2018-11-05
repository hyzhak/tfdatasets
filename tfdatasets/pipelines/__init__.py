from tfdatasets.pipelines.cifar10 import model_loader as cifar10_loader, \
    processor as cifar10_processor

from tfdatasets.pipelines import cifar10, iris

profiles = {
    'cifar10': cifar10,
    'iris': iris,
}


def get_profile(name):
    if name in profiles:
        return profiles[name]
    else:
        raise NotImplementedError()
