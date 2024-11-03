from .Kvasir_data import *
from .ISIC_data import *

datasets = {
    'kvasir': KvasirDataset,
    'isic2017': ISIC2017Dataset,
}


def get_segmentation_dataset(name, root, mode, **kwargs):
    return datasets[name.lower()](root, mode, **kwargs)
