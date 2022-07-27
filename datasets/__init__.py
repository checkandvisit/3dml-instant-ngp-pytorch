from .nsvf import NSVFDataset
from .colmap import ColmapDataset
from .instant_ngp import InstantNGPDataset


dataset_dict = {'nsvf': NSVFDataset,
                'colmap': ColmapDataset,
                'instant_ngp': InstantNGPDataset}