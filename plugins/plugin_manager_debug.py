#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
from abc import ABC
import cupy as cp
from typing import List, Dict
import importlib
import inspect
from dataclasses import dataclass
from ruamel.yaml import YAML


@dataclass
class PluginParams:
    name: str
    layer_name: str
    fill_nan: bool = False  # fill nan to invalid region
    is_height_layer: bool = False  # if this is a height layer


class PluginBase(ABC):
    """
    This is a base class of Plugins
    """

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        plugin_params : PluginParams
            The parameter of callback
        """

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
    ) -> cp.ndarray:
        """
        这将获得高程地图数据和插件层作为一个cupy数组。
        在这里运行你的处理并返回结果。
        高程地图的层数 0: 高程
                                1: 差异
                                2：是否有效
                                3：可遍历性
                                4：时间
                                5：上界
                                6: is_upper_bound
        你也可以用plugin_layers和plugin_layer_names来访问其他插件的层
        This gets the elevation map data and plugin layers as a cupy array.
        Run your processing here and return the result.
        layer of elevation_map  0: elevation
                                1: variance
                                2: is_valid
                                3: traversability
                                4: time
                                5: upper_bound
                                6: is_upper_bound
        You can also access to the other plugins' layer with plugin_layers and plugin_layer_names
        """
        pass


class PluginManger(object):
    """
    This manages the plugins.
    """

    def __init__(self, cell_n: int):
        self.cell_n = cell_n

    # 导入plugin脚本
    def init(self, plugin_params: List[PluginParams], extra_params: List[Dict]):
        self.plugin_params = plugin_params

        self.plugins = []
        for param, extra_param in zip(plugin_params, extra_params):
            # import plugins脚本文件
            m = importlib.import_module("." + param.name, package="plugins")  # -> 'module'
            # 返回由object的成员的(name,value)构成的列表
            for name, obj in inspect.getmembers(m):
                if inspect.isclass(obj) and issubclass(obj, PluginBase) and name != "PluginBase":
                    # Add cell_n to params
                    extra_param["cell_n"] = self.cell_n
                    self.plugins.append(obj(**extra_param))


        # self.plugins ：包含创建的处理类对象
        self.layers = cp.zeros((len(self.plugins), self.cell_n, self.cell_n))
        self.layer_names = self.get_layer_names()
        self.plugin_names = self.get_plugin_names()

    def load_plugin_settings(self, file_path: str):

        # 加载config/plugin_config.yaml文件
        print("Start loading plugins...")
        cfg = YAML().load(open(file_path, "r"))
        plugin_params = []
        extra_params = []
        for k, v in cfg.items():
            if v["enable"]:
                plugin_params.append(
                    PluginParams(
                        name=k if not "type" in v else v["type"],
                        layer_name=v["layer_name"],
                        fill_nan=v["fill_nan"],
                        is_height_layer=v["is_height_layer"],
                    )
                )
            extra_params.append(v["extra_params"])

        # 初始化
        self.init(plugin_params, extra_params)
        print("Loaded plugins are ", *self.plugin_names)

    def get_layer_names(self):
        names = []
        for obj in self.plugin_params:
            names.append(obj.layer_name)
        return names

    def get_plugin_names(self):
        names = []
        for obj in self.plugin_params:
            names.append(obj.name)
        return names

    def get_plugin_index_with_name(self, name: str) -> int:
        try:
            idx = self.plugin_names.index(name)
            return idx
        except Exception as e:
            print("Error with plugin {}: {}".format(name, e))
            return None

    # 获取对应层名字的idx
    def get_layer_index_with_name(self, name: str) -> int:
        try:
            idx = self.layer_names.index(name)
            return idx
        except Exception as e:
            print("Error with layer {}: {}".format(name, e))
            return None

    # 通过选定对应层的名字对其进行更新
    def update_with_name(self, name: str, elevation_map: cp.ndarray, layer_names: List[str]):
        idx = self.get_layer_index_with_name(name)
        if idx is not None:
            # 通过创建的处理类对象 对elemap进行运算
            self.layers[idx] = self.plugins[idx](elevation_map, layer_names, self.layers, self.layer_names)
    # 返回指定层名字的图层
    def get_map_with_name(self, name: str) -> cp.ndarray:
        idx = self.get_layer_index_with_name(name)
        if idx is not None:
            return self.layers[idx]

    def get_param_with_name(self, name: str) -> PluginParams:
        idx = self.get_layer_index_with_name(name)
        if idx is not None:
            return self.plugin_params[idx]


if __name__ == "__main__":
    plugins = [
        PluginParams(name="min_filter", layer_name="min_filter"),
        PluginParams(name="smooth_filter", layer_name="smooth"),
    ]
    extra_params = [{"dilation_size": 5, "iteration_n": 5}, {"input_layer_name": "elevation2"}]
    manager = PluginManger(200)
    manager.load_plugin_settings("config/plugin_config.yaml")
    print(manager.layer_names)
    print(manager.plugin_names)
    elevation_map = cp.zeros((7, 200, 200)).astype(cp.float32)
    layer_names = ["elevation", "variance", "is_valid", "traversability", "time", "upper_bound", "is_upper_bound"]
    elevation_map[0] = cp.random.randn(200, 200)
    elevation_map[2] = cp.abs(cp.random.randn(200, 200))
    # elevation_map ： (7, self.cell_n, self.cell_n)
    print("map", elevation_map[0])
    # manager.layers ：(len(self.plugins), self.cell_n, self.cell_n)
    print("layer map ", manager.layers[0])
    manager.update_with_name("min_filter", elevation_map, layer_names)
    manager.update_with_name("smooth_filter", elevation_map, layer_names)
    print(manager.get_map_with_name("smooth"))
