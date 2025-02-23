from .transforms import *
from configs.configs import get_opts, get_size


hparams = get_opts()

# get image size
size = get_size(hparams.dataset_name)

# inference_normalization
inference_transform = Compose([
    RescaleTo(size),
    ArrayToTensor(),
    Normalize()]
)
