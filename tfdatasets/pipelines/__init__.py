from tfdatasets.pipelines import cifar10


def get_pipeline_by_name(name):
    if name == 'cifar10':
        return cifar10
    else:
        raise NotImplementedError()
