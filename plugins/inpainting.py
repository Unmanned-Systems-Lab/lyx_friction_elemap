#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
from typing import List
import cupyx.scipy.ndimage as ndimage
import numpy as np
import cv2 as cv


from .plugin_manager import PluginBase


class Inpainting(PluginBase):
    """This is a filter to smoothen

    ...

    Attributes
    ----------
    cell_n: int
        width and height of the elevation map.
    """

    def __init__(self, cell_n: int = 100, method: str = "telea", **kwargs):
        super().__init__()
        if method == "telea":
            self.method = cv.INPAINT_TELEA
        elif method == "ns":  # Navier-Stokes
            self.method = cv.INPAINT_NS
        else:  # default method
            self.method = cv.INPAINT_TELEA

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
    ) -> cp.ndarray:
        # is_valid 层小于0.5的
        mask = cp.asnumpy((elevation_map[2] < 0.5).astype("uint8"))

        #是否存在有效（不需要修复）的单元格
        if (mask < 1).any():
            h = elevation_map[0]
            # 取不需要修复的单元格中的最大值和最小值
            h_max = float(h[mask < 1].max())
            h_min = float(h[mask < 1].min())
            # 编码成图像范围
            h = cp.asnumpy((elevation_map[0] - h_min) * 255 / (h_max - h_min)).astype("uint8")
            # 用邻近的像素替换那些坏标记
            # dst = cv2.inpaint（src，mask, inpaintRadius，flags）
            # src：输入8位1通道或3通道图像。
            # inpaintMask：修复掩码，8位1通道图像。非零像素表示需要修复的区域。
            # dst：输出与src具有相同大小和类型的图像。
            # inpaintRadius：算法考虑的每个点的圆形邻域的半径。
            # flags：
            #     INPAINT_NS基于Navier-Stokes的方法
            #     Alexandru Telea的INPAINT_TELEA方法
            dst = np.array(cv.inpaint(h, mask, 1, self.method))
            # 修复完后，解码成原来的范围
            h_inpainted = dst.astype(np.float32) * (h_max - h_min) / 255 + h_min
            return cp.asarray(h_inpainted).astype(np.float64)
        else:
            return elevation_map[0]
