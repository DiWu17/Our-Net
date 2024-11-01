from .Kvasir_data import *
from .ISIC_data import *

datasets = {
    'kvasir': KvasirDataset,
    'isic2017': ISIC2017Dataset,
}


def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
