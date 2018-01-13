import torchvision.models
from .wrn50_2 import wrn50_2
from .my_densenet import densenet161, densenet121, densenet169, densenet201
from .my_resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .dpn import dpn68, dpn68b, dpn92, dpn98, dpn131, dpn107
from .vgg import vgg11_bn, vgg13_bn, vgg16_bn
from .load_checkpoint import load_checkpoint


def normalizer_from_model(model_name):
    if 'inception' in model_name:
        normalizer = 'le'
    elif 'dpn' in model_name:
        normalizer = 'dpn'
    else:
        normalizer = 'torchvision'
    return normalizer


def create_model(
        model_name='resnet50',
        pretrained=False,
        in_chs=3,
        num_classes=1000,
        checkpoint_path='',
        **kwargs):

    if 'test_time_pool' in kwargs:
        test_time_pool = kwargs.pop('test_time_pool')
    else:
        test_time_pool = 0

    if model_name == 'dpn68':
        model = dpn68(
            in_chs=in_chs, num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'dpn68b':
        model = dpn68b(
            in_chs=in_chs, num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'dpn92':
        model = dpn92(
            in_chs=in_chs, num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'dpn98':
        model = dpn98(
            in_chs=in_chs, num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'dpn131':
        model = dpn131(
            in_chs=in_chs, num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'dpn107':
        model = dpn107(
            in_chs=in_chs, num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'resnet18':
        model = resnet18(in_chs=in_chs, num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet34':
        model = resnet34(in_chs=in_chs, num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet50':
        model = resnet50(in_chs=in_chs, num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet101':
        model = resnet101(in_chs=in_chs, num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet152':
        model = resnet152(in_chs=in_chs, num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet121':
        model = densenet121(in_chs=in_chs, num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet161':
        model = densenet161(in_chs=in_chs, num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet169':
        model = densenet169(in_chs=in_chs, num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet201':
        model = densenet201(in_chs=in_chs, num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'wrn50':
        model = wrn50_2(in_chs=in_chs, num_classes=num_classes, pretrained=pretrained,  **kwargs)
    elif model_name == 'vgg16':
        model = vgg16_bn(in_chs=in_chs, num_classes=num_classes)
    else:
        assert False and "Invalid model"

    if checkpoint_path and not pretrained:
        load_checkpoint(model, checkpoint_path)

    return model
