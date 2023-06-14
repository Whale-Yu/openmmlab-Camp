# 同济子豪兄 2023-2-15
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class DubaiDataset(BaseSegDataset):
    # 类别和对应的可视化配色
    METAINFO = {
        'classes':['Unlabeled','red', 'green', 'white', 'seed-black', 'seed-white'],
        'palette':[[25,25,25],[255,0,0], [0,255,0], [255,255,255], [128,128,128], [199,199,199]]
    }
    
    # 指定图像扩展名、标注扩展名
    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False, # 类别ID为0的类别是否需要除去
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)