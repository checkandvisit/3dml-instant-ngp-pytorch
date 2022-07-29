from .nsvf import NSVFDataset
from .colmap import ColmapDataset
from .instant_ngp import InstantNGPDataset
from .rtmv import RTMVDataset
from .nerfpp import NeRFPPDataset


dataset_dict = {'nsvf': NSVFDataset,
                'colmap': ColmapDataset,
                'instant_ngp': InstantNGPDataset,
                'rtmv': RTMVDataset,
                'nerfpp': NeRFPPDataset}
