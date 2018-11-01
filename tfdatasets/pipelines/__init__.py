from tfdatasets.pipelines.cifar10 import model_loader as cifar10_loader, \
    processor as cifar10_processor


def get_loader_by_name(name):
    if name == 'cifar10':
        return cifar10_loader
    else:
        raise NotImplementedError()


def get_processor_by_name(name):
    if name == 'cifar10':
        return cifar10_processor
    else:
        raise NotImplementedError()
