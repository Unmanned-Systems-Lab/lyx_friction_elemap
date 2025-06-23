

import cv2
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import numpy as np

import matplotlib.pyplot as plt
show_animation = True

import rospy
import tf


import os
import numpy as np
import threading
import subprocess


import time

from traversability_filter import get_filter_chainer, get_filter_torch
from parameter import Parameter
from custom_kernels import add_points_kernel
from custom_kernels import semantic_add_points_kernel
from custom_kernels import semantic_add_points_kernel_dynamic
from custom_kernels import error_counting_kernel
from custom_kernels import average_map_kernel
from custom_kernels import dilation_filter_kernel
from custom_kernels import normal_filter_kernel
from custom_kernels import polygon_mask_kernel
from custom_kernels import image_to_map_correspondence_kernel
from custom_kernels import color_correspondences_to_map_kernel
from custom_kernels import gray_correspondences_to_map_kernel
from custom_kernels import points_get_semantic
from custom_kernels import points_get_friction
from map_initializer import MapInitializer
from plugins.plugin_manager import PluginManger

from traversability_polygon import (
    get_masked_traversability,
    is_traversable,
    calculate_area,
    transform_to_map_position,
    transform_to_map_index,
)

import cupy as cp


import rospy
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
xp = cp
pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)


from typing import List

# from BiSeNet.tools.ht_seg_bisenet import ht_bisenet
# from lyx_deeplabv3plus import lyx_deeplabv3plus
from lyx_deeplabv3plus_fri import lyx_deeplabv3plus_fri

import copy

from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2

def creat_dir(out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)


class ElevationMap(object):
    """
    Core elevation mapping class.
    """

    def __init__(self, param: Parameter):
        self.param = param

        self.data_type: str = np.float32

        self.resolution = param.resolution
        self.center = xp.array([0, 0, 0], dtype=self.data_type)
        self.map_length = param.map_length
        # +2 is a border for outside map
        self.cell_n = int(round(self.map_length / self.resolution)) + 2

        #
        self.map_lock = threading.Lock()
        # layers: 高程,      方差,      是否有效的, 可通过性分析,    时间, 上界,         是否是上界
        # layers: elevation, variance, is_valid,   traversability, time, upper_bound, is_upper_bound
        self.elevation_map = xp.zeros((7, self.cell_n, self.cell_n), dtype=self.data_type)
        self.elevation_map_dynamic = xp.zeros((7, self.cell_n, self.cell_n), dtype=self.data_type)
        self.layer_names = [
            "elevation",
            "variance",
            "is_valid",
            "traversability",
            "time",
            "upper_bound",
            "is_upper_bound",
        ]
        # buffers
        self.traversability_buffer = xp.full((self.cell_n, self.cell_n), xp.nan)
        self.normal_map = xp.zeros((3, self.cell_n, self.cell_n), dtype=self.data_type)

        '''initial_variance: float = 10.0
        initialized_variance: float = 10.0
        '''
        # Initial variance
        self.initial_variance = param.initial_variance
        # 方差
        self.elevation_map[1] += self.initial_variance
        # 可提供性
        self.elevation_map[3] += 1.0

        # TODO 动态物体方差和通过性是不是得调整
        # 方差
        self.elevation_map_dynamic[1] += self.initial_variance
        # 可提供性
        self.elevation_map_dynamic[3] += 1.0


        # overlap clearance 重叠清除
        cell_range = int(self.param.overlap_clear_range_xy / self.resolution)
        # 将cell_range限定在（0，self.cell_n）中
        # np.clip(cell_range, min, max) -> min<=out<=max
        cell_range = np.clip(cell_range, 0, self.cell_n)
        #
        self.cell_min = self.cell_n // 2 - cell_range // 2
        self.cell_max = self.cell_n // 2 + cell_range // 2

        # Initial mean_error
        self.mean_error = 0.0
        self.additive_mean_error = 0.0

        # 编译核函数
        self.compile_kernels()

        ####可提供性部分####
        # 为什么有echo 字段
        weight_file = subprocess.getoutput('echo "' + param.weight_file + '"')

        # 导入模型权重
        param.load_weights(weight_file)

        if param.use_chainer:
            self.traversability_filter = get_filter_chainer(param.w1, param.w2, param.w3, param.w_out)
        else:
            # 创建卷积层 并导入参数
            self.traversability_filter = get_filter_torch(param.w1, param.w2, param.w3, param.w_out)
        #
        self.untraversable_polygon = xp.zeros((1, 2))

        ####自定义插件部分####
        # Plugins 插件配置和导入
        self.plugin_manager = PluginManger(cell_n=self.cell_n)
        plugin_config_file = subprocess.getoutput('echo "' + param.plugin_config_file + '"')
        self.plugin_manager.load_plugin_settings(plugin_config_file)

        # 创建地图初始化对象
        self.map_initializer = MapInitializer(self.initial_variance, param.initialized_variance, xp=cp, method="points")

        self.uv_correspondence = cp.asarray(
            np.zeros((2, self.cell_n, self.cell_n), dtype=np.float32), dtype=self.data_type,
        )
        self.valid_correspondence = cp.asarray(
            np.zeros((self.cell_n, self.cell_n), dtype=np.bool_), dtype=np.bool_
        )

        self.semantic_map = cp.zeros(
            (3, self.cell_n, self.cell_n), dtype=self.data_type,
        )
        self.new_semantic_map = cp.zeros(
            (3, self.cell_n, self.cell_n), dtype=self.data_type,
        )

        self.prior = 0.001
        
        # 改分割类别数
        ht_cls_num = 15
        self.ht_semantic_map = cp.zeros(
            (ht_cls_num, self.cell_n, self.cell_n), dtype=self.data_type,
        )
        self.ht_semantic_map +=self.prior

        # 单幀的语义结果
        self.ht_semantic_map_frame = cp.zeros(
            (ht_cls_num, self.cell_n, self.cell_n), dtype=self.data_type,
        )

        self.frame_num = 5
        self.frame_num_count = 0
        # 单幀的语义结果 累计n帧
        self.ht_semantic_map_frame_store = cp.zeros(
            (self.frame_num,ht_cls_num, self.cell_n, self.cell_n), dtype=self.data_type,
        )

        # 单幀的是否是动态物体的 贝塔分布
        self.ht_semantic_map_dynamic_frame = cp.zeros(
            (2, self.cell_n, self.cell_n), dtype=self.data_type,
        )

        #
        self.ht_semantic_map_count = cp.zeros(
            (4, self.cell_n, self.cell_n), dtype=self.data_type,
        )

        # 新增：摩擦系数地图初始化
        self.lyx_friction_map = cp.zeros((2, self.cell_n, self.cell_n), dtype=self.data_type)  # 静态物体摩擦地图
        self.lyx_friction_map_dynamic = cp.zeros((2, self.cell_n, self.cell_n), dtype=self.data_type)  # 动态物体摩擦地图
        self.lyx_friction_map_count = cp.zeros((2, self.cell_n, self.cell_n), dtype=self.data_type)  # 摩擦系数统计
        
        # 初始化新地图用于累积
        self.new_map_friction = cp.zeros((5, self.cell_n, self.cell_n), dtype=self.data_type)  # 静态摩擦新地图
        self.new_map_friction_dynamic = cp.zeros((5, self.cell_n, self.cell_n), dtype=self.data_type)  # 动态摩擦新地图



    def clear(self):
        with self.map_lock:
            self.elevation_map *= 0.0
            # Initial variance
            self.elevation_map[1] += self.initial_variance
        self.mean_error = 0.0
        self.additive_mean_error = 0.0

    def get_position(self, position):
        position[0][:] = xp.asnumpy(self.center)

    def move(self, delta_position):
        # Shift map using delta position.
        delta_position = xp.asarray(delta_position)
        delta_pixel = xp.round(delta_position[:2] / self.resolution)
        delta_position_xy = delta_pixel * self.resolution
        self.center[:2] += xp.asarray(delta_position_xy)
        self.center[2] += xp.asarray(delta_position[2])
        self.shift_map_xy(delta_pixel)
        self.shift_map_z(-delta_position[2])

    def move_to(self, position):
        # Shift map to the center of robot. 将地图移到机器人的中心
        # 高程图的中心点是基于世界坐标系的，但是里面的值都是基于自身的机器人坐标系的
        position = xp.asarray(position)
        delta = position - self.center
        # 位置改变量化处理
        delta_pixel = xp.around(delta[:2] / self.resolution)
        delta_xy = delta_pixel * self.resolution
        # 将高程图中心移动到现在的机器人位置

        # 这个self.center 是上一帧的虚拟机器人坐标系原点 在世界坐标系下的坐标 而且这个坐标的xy是分辨率的倍数
        self.center[:2] += delta_xy
        # 高程图中心所在高度对应调整
        self.center[2] += delta[2]
        # 处理之后 self.center 是当前帧的虚拟机器人坐标系原点 在世界坐标系下的坐标 而且这个坐标的xy是分辨率的倍数
        # 将高程图的值跟随机器人的移动做相应的移动
        self.shift_map_xy(-delta_pixel)
        self.shift_map_z(-delta[2])

    # 因为cp.roll会将将后面元素滚动到前面，所以对滚动元素进行置0，再将方差层置0的层调整为初始方差。
    def pad_value(self, x, shift_value, idx=None, value=0.0):
        if idx is None:
            if shift_value[0] > 0:
                x[:, : shift_value[0], :] = value
            elif shift_value[0] < 0:
                x[:, shift_value[0]:, :] = value
            if shift_value[1] > 0:
                x[:, :, : shift_value[1]] = value
            elif shift_value[1] < 0:
                x[:, :, shift_value[1]:] = value
        else:
            if shift_value[0] > 0:
                x[idx, : shift_value[0], :] = value
            elif shift_value[0] < 0:
                x[idx, shift_value[0]:, :] = value
            if shift_value[1] > 0:
                x[idx, :, : shift_value[1]] = value
            elif shift_value[1] < 0:
                x[idx, :, shift_value[1]:] = value


    def pad_value_4D(self, x, shift_value, idx=None, value=0.0):
        if idx is None:
            if shift_value[0] > 0:
                x[:,:, : shift_value[0], :] = value
            elif shift_value[0] < 0:
                x[:,:, shift_value[0]:, :] = value
            if shift_value[1] > 0:
                x[:,:, :, : shift_value[1]] = value
            elif shift_value[1] < 0:
                x[:,:, :, shift_value[1]:] = value
        else:
            if shift_value[0] > 0:
                x[:,idx, : shift_value[0], :] = value
            elif shift_value[0] < 0:
                x[:,idx, shift_value[0]:, :] = value
            if shift_value[1] > 0:
                x[:,idx, :, : shift_value[1]] = value
            elif shift_value[1] < 0:
                x[:,idx, :, shift_value[1]:] = value

    # 将高程图的值跟随机器人xy移动做相应的移动
    def shift_map_xy(self, delta_pixel):
        shift_value = delta_pixel.astype(cp.int32)
        # 机器人没有移动
        if cp.abs(shift_value).sum() == 0:
            return
        with self.map_lock:
            # 将elemap gridmap 跟随机器人的移动做相应的移动
            # self.elevation_map ：(7, self.cell_n, self.cell_n)
            self.elevation_map = cp.roll(self.elevation_map, shift_value, axis=(1, 2))
            # 因为cp.roll会将将后面元素滚动到前面，所以对滚动元素进行置0
            self.pad_value(self.elevation_map, shift_value, value=0.0)
            # 再将方差层置0的层调整为初始方差self.initial_variance：1000
            self.pad_value(self.elevation_map, shift_value, idx=1, value=self.initial_variance)

            self.elevation_map_dynamic = cp.roll(self.elevation_map_dynamic, shift_value, axis=(1, 2))
            # 因为cp.roll会将将后面元素滚动到前面，所以对滚动元素进行置0
            self.pad_value(self.elevation_map_dynamic, shift_value, value=0.0)
            # 再将方差层置0的层调整为初始方差self.initial_variance：1000
            self.pad_value(self.elevation_map_dynamic, shift_value, idx=1, value=self.initial_variance)

            # 相比与原版,可能还有很多东西
            self.semantic_map = cp.roll(self.semantic_map, shift_value, axis=(1, 2))
            # 因为cp.roll会将将后面元素滚动到前面，所以对滚动元素进行置0
            self.pad_value(self.semantic_map, shift_value, value=0.0)

            self.ht_semantic_map = cp.roll(self.ht_semantic_map, shift_value, axis=(1, 2))
            # 因为cp.roll会将将后面元素滚动到前面，所以对滚动元素进行置0
            self.pad_value(self.ht_semantic_map, shift_value, value=0.0)

            self.ht_semantic_map_frame = cp.roll(self.ht_semantic_map_frame, shift_value, axis=(1, 2))
            # 因为cp.roll会将将后面元素滚动到前面，所以对滚动元素进行置0
            self.pad_value(self.ht_semantic_map_frame, shift_value, value=0.0)

            self.ht_semantic_map_frame_store = cp.roll(self.ht_semantic_map_frame_store, shift_value, axis=(2, 3))
            # 因为cp.roll会将将后面元素滚动到前面，所以对滚动元素进行置0
            self.pad_value_4D(self.ht_semantic_map_frame_store, shift_value, value=0.0)



            self.ht_semantic_map_dynamic_frame = cp.roll(self.ht_semantic_map_dynamic_frame, shift_value, axis=(1, 2))
            # 因为cp.roll会将将后面元素滚动到前面，所以对滚动元素进行置0
            self.pad_value(self.ht_semantic_map_dynamic_frame, shift_value, value=0.0)

            self.ht_semantic_map_count = cp.roll(self.ht_semantic_map_count, shift_value, axis=(1, 2))
            # 因为cp.roll会将将后面元素滚动到前面，所以对滚动元素进行置0
            self.pad_value(self.ht_semantic_map_count, shift_value, value=0.0)

            # 新增：摩擦系数地图的移动处理
            self.lyx_friction_map = cp.roll(self.lyx_friction_map, shift_value, axis=(1, 2))
            self.pad_value(self.lyx_friction_map, shift_value, value=0.0)
            
            self.lyx_friction_map_dynamic = cp.roll(self.lyx_friction_map_dynamic, shift_value, axis=(1, 2))
            self.pad_value(self.lyx_friction_map_dynamic, shift_value, value=0.0)
            
            self.lyx_friction_map_count = cp.roll(self.lyx_friction_map_count, shift_value, axis=(1, 2))
            self.pad_value(self.lyx_friction_map_count, shift_value, value=0.0)


    # 将高程图的值(高度值、上界值)跟随机器人z移动做相应的移动
    def shift_map_z(self, delta_z):
        with self.map_lock:
            # elevation
            self.elevation_map[0] += delta_z
            # upper bound
            self.elevation_map[5] += delta_z

            # elevation
            self.elevation_map_dynamic[0] += delta_z
            # upper bound
            self.elevation_map_dynamic[5] += delta_z

    # 编译各核函数
    def compile_kernels(self):
        # Compile custom cuda kernels.
        self.new_map = cp.zeros((7, self.cell_n, self.cell_n), dtype=self.data_type)
        self.new_map_dynamic = cp.zeros((7, self.cell_n, self.cell_n), dtype=self.data_type)

        self.traversability_input = cp.zeros((self.cell_n, self.cell_n), dtype=self.data_type)
        self.traversability_mask_dummy = cp.zeros((self.cell_n, self.cell_n), dtype=self.data_type)

        self.min_filtered = cp.zeros((self.cell_n, self.cell_n), dtype=self.data_type)
        self.min_filtered_mask = cp.zeros((self.cell_n, self.cell_n), dtype=self.data_type)

        self.mask = cp.zeros((self.cell_n, self.cell_n), dtype=self.data_type)

        # 对点云进行过滤 光线追踪 上边界 等一系列操作
        self.add_points_kernel = add_points_kernel(
            self.resolution,
            self.cell_n,
            self.cell_n,
            self.param.sensor_noise_factor,
            self.param.mahalanobis_thresh,
            self.param.outlier_variance,
            self.param.wall_num_thresh,
            self.param.max_ray_length,
            self.param.cleanup_step,
            self.param.min_valid_distance,
            self.param.max_height_range,
            self.param.cleanup_cos_thresh,
            self.param.ramped_height_range_a,
            self.param.ramped_height_range_b,
            self.param.ramped_height_range_c,
            self.param.enable_edge_sharpen,
            self.param.enable_visibility_cleanup,
        )

        self.semantic_add_points_kernel = semantic_add_points_kernel(
            self.resolution,
            self.cell_n,
            self.cell_n,
            self.param.sensor_noise_factor,
            self.param.mahalanobis_thresh,
            self.param.outlier_variance,
            self.param.wall_num_thresh,
            self.param.max_ray_length,
            self.param.cleanup_step,
            self.param.min_valid_distance,
            self.param.max_height_range,
            self.param.cleanup_cos_thresh,
            self.param.ramped_height_range_a,
            self.param.ramped_height_range_b,
            self.param.ramped_height_range_c,
            self.param.enable_edge_sharpen,
            self.param.enable_visibility_cleanup,
        )

        self.semantic_add_points_kernel_dynamic = semantic_add_points_kernel_dynamic(
            self.resolution,
            self.cell_n,
            self.cell_n,
            self.param.sensor_noise_factor,
            self.param.mahalanobis_thresh,
            self.param.outlier_variance,
            self.param.wall_num_thresh,
            self.param.max_ray_length,
            self.param.cleanup_step,
            self.param.min_valid_distance,
            self.param.max_height_range,
            self.param.cleanup_cos_thresh,
            self.param.ramped_height_range_a,
            self.param.ramped_height_range_b,
            self.param.ramped_height_range_c,
            self.param.enable_edge_sharpen,
            self.param.enable_visibility_cleanup,
        )


        # 误差计算、累计、计数
        self.error_counting_kernel = error_counting_kernel(
            self.resolution,
            self.cell_n,
            self.cell_n,
            self.param.sensor_noise_factor,
            self.param.mahalanobis_thresh,
            self.param.drift_compensation_variance_inlier,
            self.param.traversability_inlier,
            self.param.min_valid_distance,
            self.param.max_height_range,
            self.param.ramped_height_range_a,
            self.param.ramped_height_range_b,
            self.param.ramped_height_range_c,
        )

        # 平均输入的gridmap，判断、计算高度和方差
        self.average_map_kernel = average_map_kernel(
            self.cell_n, self.cell_n, self.param.max_variance, self.initial_variance
        )

        # 在高度图上，对无效的单元格进行扩张，取为最近的有效单元的值
        self.dilation_filter_kernel = dilation_filter_kernel(self.cell_n, self.cell_n, self.param.dilation_size)
        self.dilation_filter_kernel_initializer = dilation_filter_kernel(
            self.cell_n, self.cell_n, self.param.dilation_size_initialize
        )

        #
        self.polygon_mask_kernel = polygon_mask_kernel(self.cell_n, self.cell_n, self.resolution)

        # 求有效单元格的法向量
        self.normal_filter_kernel = normal_filter_kernel(self.cell_n, self.cell_n, self.resolution)

        self.image_to_map_correspondence_kernel = image_to_map_correspondence_kernel(
            resolution=self.resolution, width=self.cell_n, height=self.cell_n, tolerance_z_collision=0.10,
        )
        # self.color_correspondences_to_map_kernel = color_correspondences_to_map_kernel(
        #     resolution=self.resolution, width=self.cell_n, height=self.cell_n,
        # )

        self.gray_correspondences_to_map_kernel = gray_correspondences_to_map_kernel(
            resolution=self.resolution, width=self.cell_n, height=self.cell_n,
        )

    def shift_translation_to_map_center(self, t):
        # self.center 是当前帧的机器人坐标系原点 在地图坐标系的坐标
        # t是 当前帧 传感器坐标系原点 在地图坐标系的坐标
        t -= self.center

    # 使用核函数计算 更新map
    def update_map_with_kernel(self, points, R, t, position_noise, orientation_noise):
        # self.new_map ：(7, self.cell_n, self.cell_n)
        self.new_map *= 0.0

        error = cp.array([0.0], dtype=cp.float32)
        error_cnt = cp.array([0], dtype=cp.float32)
        with self.map_lock:
            # self.center是机器人中心在map坐标系的表示
            # 得到传感器坐标系原点在机器人坐标系的坐标
            self.shift_translation_to_map_center(t)

            # 对筛选出的“非常”有效的单元格进行误差计算、累计、计数，可穿越性设置为1
            self.error_counting_kernel(
                # 输入
                self.elevation_map,
                points,
                cp.array([0.0], dtype=self.data_type),
                cp.array([0.0], dtype=self.data_type),
                R,  # 虚拟机器人坐标系（与机器人坐标系同原点，但是坐标轴与地图坐标系一样）传感坐标系在 地图坐标系到传感坐标系的变换
                t,  # 传感器坐标系原点在虚拟机器人坐标系的表示 传感器坐标系原点在机器人坐标系的表示

                # 输出
                self.new_map,  # 累加"traversability"和"time",
                error,  # 输入点云与输入elemap的高度的偏差累计量 对筛选出的“非常”有效的单元格 当前帧与上一帧对应单元格的偏差累计量
                error_cnt,  # 偏差累计量所包含的点数量

                #
                size=(points.shape[0]),
            )

            # 如果筛选出的“非常”有效的单元格大于一定数量，并且位置和角度偏差都大于了一定阈值，就进行漂移补偿
            # 这种补偿只是平均的高度方向的补偿
            if (
                    self.param.enable_drift_compensation
                    # 最小高度漂移补偿的有效单元格数量 漂移补偿只发生在有效单元格超过这个数字的时候
                    # min_height_drift_cnt: 100                       # drift compensation only happens if the valid cells are more than this number.
                    and error_cnt > self.param.min_height_drift_cnt
                    and (
                    # 位置偏差/噪声阈值 如果位置变化大于此值，则发生漂移补偿。
                    # position_noise_thresh: 0.01                     # if the position change is bigger than this value, the drift compensation happens.
                    # 方向偏差/噪声阈值 如果方向变化大于此值，则发生漂移补偿
                    # orientation_noise_thresh: 0.01                  # if the orientation change is bigger than this value, the drift compensation happens.
                    position_noise > self.param.position_noise_thresh
                    or orientation_noise > self.param.orientation_noise_thresh
            )
            ):
                # 平均高度偏差
                self.mean_error = error / error_cnt
                #
                self.additive_mean_error += self.mean_error
                # 最大漂移 漂移补偿只有在漂移小于这个值时才会发生（为了安全）。
                if np.abs(self.mean_error) < self.param.max_drift:
                    # 根据偏差值更新elemap高度
                    # 漂移补偿α，使漂移补偿的更新更加平滑
                    # drift_compensation_alpha: 0.1                   # drift compensation alpha for smoother update of drift compensation
                    self.elevation_map[0] += self.mean_error * self.param.drift_compensation_alpha

            # 对点云进行过滤 更新高度及方差 光线追踪 上边界 等一系列操作
            self.add_points_kernel(
                # 输入
                points,
                cp.array([0.0], dtype=self.data_type),
                cp.array([0.0], dtype=self.data_type),
                R,
                t,
                self.normal_map,

                # 输出
                self.elevation_map,  # 更新"variance"，"is_valid","time","upper_bound","is_upper_bound",
                self.new_map,  # 累加"elevation","variance","is_valid",
                size=(points.shape[0]),
            )

            # 平均输入的self.new_map，判断、计算高度和方差 更新到self.elevation_map
            self.average_map_kernel(self.new_map,

                                    self.elevation_map,  # 更新 "elevation","variance","is_valid"

                                    size=(self.cell_n * self.cell_n))

            # 重叠部分消除，具体地，消除指定机器人附近xy的在机器人所在高度+h，-h以外的单元格的"elevation","variance","is_valid"
            if self.param.enable_overlap_clearance:
                self.clear_overlap_map(t)

            # dilation before traversability_filter 可穿越性过滤器前的扩张
            # (self.cell_n, self.cell_n)
            self.traversability_input *= 0.0

            # 在高度图上，对无效的单元格进行扩张，取为最近的有效单元的值
            self.dilation_filter_kernel(
                self.elevation_map[5],
                self.elevation_map[2] + self.elevation_map[6],

                self.traversability_input,  # "elevation"层，在指定扩张size范围内对无效的单元格取为最近的单元格，扩张的单元格的valid没有置1
                self.traversability_mask_dummy,  # 假的alid值，只有扩张的单元格的valid为1

                size=(self.cell_n * self.cell_n),
            )

            # calculate traversability
            traversability = self.traversability_filter(self.traversability_input)
            # self.elevation_map[3] ："traversability" ( self.cell_n-6，self.cell_n-6 )
            self.elevation_map[3][3:-3, 3:-3] = traversability.reshape(
                (traversability.shape[2], traversability.shape[3])
            )

        # calculate normal vectors
        #
        self.update_normal(self.traversability_input)

    # 使用核函数计算 更新map
    def update_map_with_kernel_semantic_points(self, points, R, t, position_noise, orientation_noise):
        # 查看为nan的值
        # ls_elevation_map = self.elevation_map.copy()
        # ls_elevation_map = cp.asnumpy(ls_elevation_map)
        # nan_indices = np.argwhere(np.isnan(ls_elevation_map[0]))
        # for nan_indices_ele in nan_indices:
        #     print('nan_indices_ele:',nan_indices_ele)
        #     print('variance :',ls_elevation_map[1,nan_indices_ele[0],nan_indices_ele[1]])
        #     # print('variance :', )
        # print('variance :', ls_elevation_map[1, 910, 423])
        # print('ele :', ls_elevation_map[0, 910, 423])


        # self.new_map ：(7, self.cell_n, self.cell_n)
        self.new_map *= 0.0
        self.new_map_dynamic *= 0.0
        self.ht_semantic_map_frame *=0.0
        self.new_map_friction *= 0.0  # 新增：重置摩擦累积地图
        self.new_map_friction_dynamic *= 0.0  # 新增：重置动态摩擦累积地图
        # self.ht_semantic_map *= 0
        error = cp.array([0.0], dtype=cp.float32)
        error_cnt = cp.array([0], dtype=cp.float32)
        with self.map_lock:
            # self.center是机器人中心在map坐标系的表示
            # 得到传感器坐标系原点在机器人坐标系的坐标
            self.shift_translation_to_map_center(t)

            # 对筛选出的“非常”有效的单元格进行误差计算、累计、计数，可穿越性设置为1
            self.error_counting_kernel(
                # 输入
                self.elevation_map,
                points,
                cp.array([0.0], dtype=self.data_type),
                cp.array([0.0], dtype=self.data_type),
                R,  # 虚拟机器人坐标系（与机器人坐标系同原点，但是坐标轴与地图坐标系一样）传感坐标系在 地图坐标系到传感坐标系的变换
                t,  # 传感器坐标系原点在虚拟机器人坐标系的表示 传感器坐标系原点在机器人坐标系的表示

                # 输出
                self.new_map,  # 累加"traversability"和"time",
                error,  # 输入点云与输入elemap的高度的偏差累计量 对筛选出的“非常”有效的单元格 当前帧与上一帧对应单元格的偏差累计量
                error_cnt,  # 偏差累计量所包含的点数量

                #
                size=(points.shape[0]),
            )

            # 如果筛选出的“非常”有效的单元格大于一定数量，并且位置和角度偏差都大于了一定阈值，就进行漂移补偿
            # 这种补偿只是平均的高度方向的补偿
            if (
                    self.param.enable_drift_compensation
                    # 最小高度漂移补偿的有效单元格数量 漂移补偿只发生在有效单元格超过这个数字的时候
                    # min_height_drift_cnt: 100                       # drift compensation only happens if the valid cells are more than this number.
                    and error_cnt > self.param.min_height_drift_cnt
                    and (
                    # 位置偏差/噪声阈值 如果位置变化大于此值，则发生漂移补偿。
                    # position_noise_thresh: 0.01                     # if the position change is bigger than this value, the drift compensation happens.
                    # 方向偏差/噪声阈值 如果方向变化大于此值，则发生漂移补偿
                    # orientation_noise_thresh: 0.01                  # if the orientation change is bigger than this value, the drift compensation happens.
                    position_noise > self.param.position_noise_thresh
                    or orientation_noise > self.param.orientation_noise_thresh
            )
            ):
                # 平均高度偏差
                self.mean_error = error / error_cnt
                #
                self.additive_mean_error += self.mean_error
                # 最大漂移 漂移补偿只有在漂移小于这个值时才会发生（为了安全）。
                if np.abs(self.mean_error) < self.param.max_drift:
                    # 根据偏差值更新elemap高度
                    # 漂移补偿α，使漂移补偿的更新更加平滑
                    # drift_compensation_alpha: 0.1                   # drift compensation alpha for smoother update of drift compensation
                    self.elevation_map[0] += self.mean_error * self.param.drift_compensation_alpha
            # print("points[:, 3].max()",points[:, 3].max())

            # # 对点云进行过滤 更新高度及方差 光线追踪 上边界 等一系列操作
            # self.semantic_add_points_kernel(
            #     # 输入
            #     points,
            #     cp.array([0.0], dtype=self.data_type),
            #     cp.array([0.0], dtype=self.data_type),
            #     R,
            #     t,
            #     self.normal_map,
            #
            #     # 输出
            #     self.elevation_map,  # 更新"variance"，"is_valid","time","upper_bound","is_upper_bound",
            #     self.new_map,  # 累加"elevation","variance","is_valid",
            #
            #     self.ht_semantic_map_frame,
            #     size=(points.shape[0]),
            # )

            # print(f"调用核函数前检查:")
            # print(f"points形状: {points.shape}")
            # print(f"摩擦地图形状: {self.lyx_friction_map.shape}")
            # print(f"动态摩擦地图形状: {self.lyx_friction_map_dynamic.shape}")
            # 区分静态和动态物体
            # 对点云进行过滤 更新高度及方差 光线追踪 上边界 等一系列操作
            self.semantic_add_points_kernel_dynamic(
                # 输入
                points,
                cp.array([0.0], dtype=self.data_type),
                cp.array([0.0], dtype=self.data_type),
                R,
                t,
                self.normal_map,

                # 输出
                self.elevation_map,  # 更新"variance"，"is_valid","time","upper_bound","is_upper_bound",
                self.new_map,  # 累加"elevation","variance","is_valid",
                self.elevation_map_dynamic,  # 更新"variance"，"is_valid","time","upper_bound","is_upper_bound",
                self.new_map_dynamic,  # 累加"elevation","variance","is_valid",

                self.ht_semantic_map_frame,
                self.lyx_friction_map,  # 新增：静态摩擦地图
                self.lyx_friction_map_dynamic,  # 新增：动态摩擦地图
                
                size=(points.shape[0]),
            )


            self.ht_semantic_map +=self.ht_semantic_map_frame
            self.ht_semantic_map_frame_store[0,:,:,:] = self.ht_semantic_map_frame
            if self.frame_num_count< self.frame_num:
                self.frame_num_count +=1
            self.ht_semantic_map_frame_store_sum = np.sum(self.ht_semantic_map_frame_store[0:self.frame_num_count,:, :, :], axis=0)
            # self.ht_semantic_map_dynamic_frame *= 0
            # # 动态物体  使用多帧的结果
            # self.ht_semantic_map_dynamic_frame[0,:,:] = np.sum(self.ht_semantic_map_frame_store_sum[0:11, :, :], axis=0)+0.001
            # self.ht_semantic_map_dynamic_frame[1, :, :] = np.sum(self.ht_semantic_map_frame_store_sum[11:, :, :], axis=0)
            
            # dynamic_indices = [12, 15]  # 只有vehicle和object视为动态
            # static_indices = [i for i in range(18) if i not in dynamic_indices]

            dynamic_indices = []  # 空列表 - 没有动态物体
            static_indices = list(range(15))  # 所有类别都是静态的            

            dynamic_sum = np.zeros_like(self.ht_semantic_map_frame_store_sum[0,:,:])
            for idx in dynamic_indices:
                dynamic_sum += self.ht_semantic_map_frame_store_sum[idx,:,:]

            static_sum = np.zeros_like(self.ht_semantic_map_frame_store_sum[0,:,:])
            for idx in static_indices:
                static_sum += self.ht_semantic_map_frame_store_sum[idx,:,:]

            self.ht_semantic_map_dynamic_frame[0,:,:] = static_sum + 0.001
            self.ht_semantic_map_dynamic_frame[1,:,:] = dynamic_sum
            self.ht_semantic_map_frame_store = cp.roll(self.ht_semantic_map_frame_store, 1, axis=0)


            # # self.ht_semantic_map_dynamic_frame *= 0
            # # 动态物体
            # self.ht_semantic_map_dynamic_frame[0,:,:] = np.sum(self.ht_semantic_map_frame[0:11, :, :], axis=0)+0.001
            # self.ht_semantic_map_dynamic_frame[1, :, :] = np.sum(self.ht_semantic_map_frame[11:, :, :], axis=0)

            # 数值最大的索引
            self.ht_semantic_map_count[0,:,:] = cp.argmax(self.ht_semantic_map, axis=0)
            # 数值最大的值
            self.ht_semantic_map_count[1, :, :] = cp.max(self.ht_semantic_map, axis=0)
            # 动态 数值最大的索引
            self.ht_semantic_map_count[2, :, :] = cp.argmax(self.ht_semantic_map_dynamic_frame, axis=0)
            # 动态 数值最大的值
            self.ht_semantic_map_count[3, :, :] = self.ht_semantic_map_dynamic_frame[1, :, :]


            # 平均输入的self.new_map，判断、计算高度和方差 更新到self.elevation_map
            self.average_map_kernel(self.new_map,

                                    self.elevation_map,  # 更新 "elevation","variance","is_valid"

                                    size=(self.cell_n * self.cell_n))

            # 平均输入的self.new_map，判断、计算高度和方差 更新到self.elevation_map
            self.average_map_kernel(self.new_map_dynamic,

                                    self.elevation_map_dynamic,  # 更新 "elevation","variance","is_valid"

                                    size=(self.cell_n * self.cell_n))
            
            # # 对新累积的摩擦地图进行平均处理，与semantic相同的方式
            # self.average_map_kernel(self.new_map_friction,
            #                         self.lyx_friction_map,  # 更新摩擦地图
            #                         size=(self.cell_n * self.cell_n))
            
            # self.average_map_kernel(self.new_map_friction_dynamic,
            #                         self.lyx_friction_map_dynamic,  # 更新动态摩擦地图
            #                         size=(self.cell_n * self.cell_n))

            # 重叠部分消除，具体地，消除指定机器人附近xy的在机器人所在高度+h，-h以外的单元格的"elevation","variance","is_valid"
            if self.param.enable_overlap_clearance:
                self.clear_overlap_map(t)

            # dilation before traversability_filter 可穿越性过滤器前的扩张
            # (self.cell_n, self.cell_n)
            self.traversability_input *= 0.0

            # # 在高度图上，对无效的单元格进行扩张，取为最近的有效单元的值
            # self.dilation_filter_kernel(
            #     self.elevation_map[5],
            #     self.elevation_map[2] + self.elevation_map[6],
            #
            #     self.traversability_input,  # "elevation"层，在指定扩张size范围内对无效的单元格取为最近的单元格，扩张的单元格的valid没有置1
            #     self.traversability_mask_dummy,  # 假的alid值，只有扩张的单元格的valid为1
            #
            #     size=(self.cell_n * self.cell_n),
            # )

            # calculate traversability
            traversability = self.traversability_filter(self.traversability_input)
            # self.elevation_map[3] ："traversability" ( self.cell_n-6，self.cell_n-6 )
            self.elevation_map[3][3:-3, 3:-3] = traversability.reshape(
                (traversability.shape[2], traversability.shape[3])
            )

        # calculate normal vectors
        #
        self.update_normal(self.traversability_input)


    # 重叠部分消除，具体地，消除指定机器人附近xy的在机器人所在高度+h，-h以外的单元格的"elevation","variance","is_valid"
    def clear_overlap_map(self, t):
        # Clear overlapping area around center 清除机器人中心周围的重叠部分
        height_min = t[2] - self.param.overlap_clear_range_z
        height_max = t[2] + self.param.overlap_clear_range_z

        near_map = self.elevation_map[:, self.cell_min: self.cell_max, self.cell_min: self.cell_max]

        valid_idx = ~cp.logical_or(near_map[0] < height_min, near_map[0] > height_max)
        near_map[0] = cp.where(valid_idx, near_map[0], 0.0)
        near_map[1] = cp.where(valid_idx, near_map[1], self.initial_variance)
        near_map[2] = cp.where(valid_idx, near_map[2], 0.0)

        valid_idx = ~cp.logical_or(near_map[5] < height_min, near_map[5] > height_max)
        near_map[5] = cp.where(valid_idx, near_map[5], 0.0)
        near_map[6] = cp.where(valid_idx, near_map[6], 0.0)

        self.elevation_map[:, self.cell_min: self.cell_max, self.cell_min: self.cell_max] = near_map

    def get_additive_mean_error(self):
        return self.additive_mean_error

    # 更新方差层 时间方差，以恒定的速度增加方差
    def update_variance(self):
        # 为什么要乘有效值
        self.elevation_map[1] += self.param.time_variance * self.elevation_map[2]

    # 更新时间层
    def update_time(self):
        self.elevation_map[4] += self.param.time_interval

    def update_upper_bound_with_valid_elevation(self):
        mask = self.elevation_map[2] > 0.5
        self.elevation_map[5] = cp.where(mask, self.elevation_map[0], self.elevation_map[5])
        self.elevation_map[6] = cp.where(mask, 0.0, self.elevation_map[6])

    def input(self, raw_points, R, t, position_noise, orientation_noise):
        # Update elevation map using point cloud input.
        raw_points = cp.asarray(raw_points, dtype=self.data_type)
        # 过滤掉 nan
        raw_points = raw_points[~cp.isnan(raw_points).any(axis=1)]
        # 计算了可通过性地图
        self.update_map_with_kernel(raw_points, cp.asarray(R, dtype=self.data_type), cp.asarray(t, dtype=self.data_type), position_noise, orientation_noise)

    def input_semantic_points(self, raw_points, R, t, position_noise, orientation_noise):
        # Update elevation map using point cloud input.
        raw_points = cp.asarray(raw_points, dtype=self.data_type)
        # 过滤掉 nan
        raw_points = raw_points[~cp.isnan(raw_points).any(axis=1)]
        # 计算了可通过性地图
        self.update_map_with_kernel_semantic_points(raw_points, cp.asarray(R, dtype=self.data_type), cp.asarray(t, dtype=self.data_type), position_noise, orientation_noise)


    def update_normal(self, dilated_map):
        # self.traversability_input 扩张了之后的elevation层
        with self.map_lock:
            self.normal_map *= 0.0
            # 求有效单元格的法向量
            self.normal_filter_kernel(
                dilated_map,
                self.elevation_map[2],  # 扩张的单元格的valid没有置1

                self.normal_map,

                size=(self.cell_n * self.cell_n)
            )

    def process_map_for_publish(self, input_map, fill_nan=False, add_z=False, xp=cp):
        m = input_map.copy()
        if fill_nan:
            m = xp.where(self.elevation_map[2] > 0.5, m, xp.nan)
        if add_z:
            m = m + self.center[2]
        return m[1:-1, 1:-1]
    def process_map_for_publish_zerofill(self, input_map, fill_nan=False, add_z=False, xp=cp):
        m = input_map.copy()
        if fill_nan:
            m = xp.where(self.elevation_map[2] > 0.5, m, xp.nan)
            # m = xp.where(self.elevation_map[2] > 0.5, m, -100)
        if add_z:
            m = m + self.center[2]
        # print('xp.max(m):',xp.max(m))
        # print('xp.min(m):', xp.min(m))



        return m[1:-1, 1:-1]

    def process_map_for_publish_zerofill_dynamic(self, input_map, fill_nan=False, add_z=False, xp=cp):
        m = input_map.copy()
        if fill_nan:
            m = xp.where(self.elevation_map_dynamic[2] > 0.5, m, -0.3)
        if add_z:
            m = m + self.center[2]
        return m[1:-1, 1:-1]


    def get_elevation(self):
        return self.process_map_for_publish_zerofill(self.elevation_map[0], fill_nan=True, add_z=False)

    def get_elevation_dynamic(self):
        return self.process_map_for_publish_zerofill_dynamic(self.elevation_map_dynamic[0], fill_nan=True, add_z=False)

    def get_variance(self):
        return self.process_map_for_publish(self.elevation_map[1], fill_nan=False, add_z=False)

    def get_traversability(self):
        traversability = cp.where(
            (self.elevation_map[2] + self.elevation_map[6]) > 0.5, self.elevation_map[3].copy(), cp.nan
        )
        self.traversability_buffer[3:-3, 3:-3] = traversability[3:-3, 3:-3]
        traversability = self.traversability_buffer[1:-1, 1:-1]
        return traversability

    def get_time(self):
        return self.process_map_for_publish(self.elevation_map[4], fill_nan=False, add_z=False)

    def get_upper_bound(self):
        if self.param.use_only_above_for_upper_bound:
            valid = cp.logical_or(
                cp.logical_and(self.elevation_map[5] > 0.0, self.elevation_map[6] > 0.5), self.elevation_map[2] > 0.5
            )
        else:
            valid = cp.logical_or(self.elevation_map[2] > 0.5, self.elevation_map[6] > 0.5)
        upper_bound = cp.where(valid, self.elevation_map[5].copy(), cp.nan)
        upper_bound = upper_bound[1:-1, 1:-1] + self.center[2]
        return upper_bound

    def get_is_upper_bound(self):
        if self.param.use_only_above_for_upper_bound:
            valid = cp.logical_or(
                cp.logical_and(self.elevation_map[5] > 0.0, self.elevation_map[6] > 0.5), self.elevation_map[2] > 0.5
            )
        else:
            valid = cp.logical_or(self.elevation_map[2] > 0.5, self.elevation_map[6] > 0.5)
        is_upper_bound = cp.where(valid, self.elevation_map[6].copy(), cp.nan)
        is_upper_bound = is_upper_bound[1:-1, 1:-1]
        return is_upper_bound

    def xp_of_array(self, array):
        if type(array) == cp.ndarray:
            return cp
        elif type(array) == np.ndarray:
            return np

    def copy_to_cpu(self, array, data, stream=None):
        if type(array) == np.ndarray:
            data[...] = array.astype(np.float32)
        elif type(array) == cp.ndarray:
            if stream is not None:
                data[...] = cp.asnumpy(array.astype(np.float32), stream=stream)
            else:
                data[...] = cp.asnumpy(array.astype(np.float32))

    def exists_layer(self, name):
        if name in self.layer_names:
            return True
        elif name in self.plugin_manager.layer_names:
            return True
        else:
            return False

    def get_map_with_name_ref(self, name, data):
        use_stream = True
        xp = cp
        with self.map_lock:
            if name == "elevation":
                m = self.get_elevation()
                use_stream = False
            elif name == "elevation_dynamic":
                m = self.get_elevation_dynamic()
            elif name == "variance":
                m = self.get_variance()
            elif name == "traversability":
                m = self.get_traversability()
            elif name == "time":
                m = self.get_time()
            elif name == "upper_bound":
                m = self.get_upper_bound()
            elif name == "is_upper_bound":
                m = self.get_is_upper_bound()
            elif name == "normal_x":
                m = self.normal_map.copy()[0, 1:-1, 1:-1]
            elif name == "normal_y":
                m = self.normal_map.copy()[1, 1:-1, 1:-1]
            elif name == "normal_z":
                m = self.normal_map.copy()[2, 1:-1, 1:-1]
            elif name == "uv_correspondence_u":
                m = self.uv_correspondence.copy()[0, 1:-1, 1:-1]
            elif name == "uv_correspondence_v":
                m = self.uv_correspondence.copy()[1, 1:-1, 1:-1]
            elif name == "semantic_map":
                m = self.semantic_map.copy()[0, 1:-1, 1:-1]
            elif name in self.plugin_manager.layer_names:
                self.plugin_manager.update_with_name(name, self.elevation_map, self.layer_names)
                m = self.plugin_manager.get_map_with_name(name)
                p = self.plugin_manager.get_param_with_name(name)
                xp = self.xp_of_array(m)
                m = self.process_map_for_publish(m, fill_nan=p.fill_nan, add_z=p.is_height_layer, xp=xp)
            else:
                # print("Layer {} is not in the map".format(name))
                return
        m = xp.flip(m, 0)
        m = xp.flip(m, 1)
        if use_stream:
            stream = cp.cuda.Stream(non_blocking=False)
        else:
            stream = None
        self.copy_to_cpu(m, data, stream=stream)


    def get_normal_maps(self):
        normal = self.normal_map.copy()
        normal_x = normal[0, 1:-1, 1:-1]
        normal_y = normal[1, 1:-1, 1:-1]
        normal_z = normal[2, 1:-1, 1:-1]
        maps = xp.stack([normal_x, normal_y, normal_z], axis=0)
        maps = xp.flip(maps, 1)
        maps = xp.flip(maps, 2)
        maps = xp.asnumpy(maps)
        return maps

    def get_normal_ref(self, normal_x_data, normal_y_data, normal_z_data):
        maps = self.get_normal_maps()
        self.stream = cp.cuda.Stream(non_blocking=True)
        normal_x_data[...] = xp.asnumpy(maps[0], stream=self.stream)
        normal_y_data[...] = xp.asnumpy(maps[1], stream=self.stream)
        normal_z_data[...] = xp.asnumpy(maps[2], stream=self.stream)

    def get_polygon_traversability(self, polygon, result):
        polygon = xp.asarray(polygon)
        area = calculate_area(polygon)
        pmin = self.center[:2] - self.map_length / 2 + self.resolution
        pmax = self.center[:2] + self.map_length / 2 - self.resolution
        polygon[:, 0] = polygon[:, 0].clip(pmin[0], pmax[0])
        polygon[:, 1] = polygon[:, 1].clip(pmin[1], pmax[1])
        polygon_min = polygon.min(axis=0)
        polygon_max = polygon.max(axis=0)
        polygon_bbox = cp.concatenate([polygon_min, polygon_max]).flatten()
        polygon_n = polygon.shape[0]
        clipped_area = calculate_area(polygon)
        self.polygon_mask_kernel(
            polygon,
            self.center[0],
            self.center[1],
            polygon_n,
            polygon_bbox,
            self.mask,
            size=(self.cell_n * self.cell_n),
        )
        masked, masked_isvalid = get_masked_traversability(self.elevation_map, self.mask)
        if masked_isvalid.sum() > 0:
            t = masked.sum() / masked_isvalid.sum()
        else:
            t = 0.0
        is_safe, un_polygon = is_traversable(
            masked, self.param.safe_thresh, self.param.safe_min_thresh, self.param.max_unsafe_n
        )
        untraversable_polygon_num = 0
        if un_polygon is not None:
            un_polygon = transform_to_map_position(un_polygon, self.center[:2], self.cell_n, self.resolution)
            untraversable_polygon_num = un_polygon.shape[0]
        if clipped_area < 0.001:
            is_safe = False
            print("requested polygon is outside of the map")
        result[...] = np.array([is_safe, t, area])
        self.untraversable_polygon = un_polygon
        return untraversable_polygon_num

    def get_untraversable_polygon(self, untraversable_polygon):
        untraversable_polygon[...] = xp.asnumpy(self.untraversable_polygon)

    def initialize_map(self, points, method="cubic"):
        # elevation_map 置为0，方差层设置为初始方差
        # mean_error additive_mean_error置为0
        self.clear()
        with self.map_lock:
            points = cp.asarray(points, dtype=self.data_type)
            indices = transform_to_map_index(points[:, :2], self.center[:2], self.cell_n, self.resolution)
            points[:, :2] = indices.astype(points.dtype)
            points[:, 2] -= self.center[2]
            self.map_initializer(self.elevation_map, points, method)
            if self.param.dilation_size_initialize > 0:
                for i in range(2):
                    self.dilation_filter_kernel_initializer(
                        self.elevation_map[0],
                        self.elevation_map[2],
                        self.elevation_map[0],
                        self.elevation_map[2],
                        size=(self.cell_n * self.cell_n),
                    )
            self.update_upper_bound_with_valid_elevation()

    def input_image(
        self,
        # 检查多通道图像
        image: List[cp._core.core.ndarray],
        # 通道类型
        channels: List[str],
        # fusion_methods: List[str],
        R: cp._core.core.ndarray,
        t: cp._core.core.ndarray,
        # 内参矩阵
        K: cp._core.core.ndarray,
        image_height: int,
        image_width: int,
    ):
        """Input image and fuse the new measurements to update the elevation map.
        # 输入图像并融合新的测量结果，更新高程图。

        Args:
            sub_key (str): Key used to identify the subscriber configuration
            image (List[cupy._core.core.ndarray]): List of array containing the individual image input channels
            R (cupy._core.core.ndarray): Camera optical center rotation
            t (cupy._core.core.ndarray): Camera optical center translation
            K (cupy._core.core.ndarray): Camera intrinsics
            image_height (int): Image height
            image_width (int): Image width

        Returns:
            None:

        参数：
            sub_key（str）： 用于标识订阅者配置的密钥
            image（List[cupy._core.core.ndarray]）： 包含单个图像输入通道的数组列表
            R（cupy._core.core.ndarray）： 相机光学中心旋转
            t （cupy._core.core.ndarray）： 摄像机光学中心平移： 摄像机光学中心平移
            K (cupy._core.core.ndarray)： 摄像机本征
            图像高度（int）： 图像高度
            image_width (int)： 图像宽度

        返回 返回 返回 返回值值值值
            无：

        """
        image = np.stack(image, axis=0)
        if len(image.shape) == 2:
            # 扩充一个维度
            image = image[None]

        # Convert to cupy
        image = cp.asarray(image, dtype=self.data_type)
        K = cp.asarray(K, dtype=self.data_type)
        # R_zed_from_map
        R = cp.asarray(R, dtype=self.data_type)
        t = cp.asarray(t, dtype=self.data_type)
        image_height = cp.float32(image_height)
        image_width = cp.float32(image_width)

        # Calculate transformation matrix
        # TODO
        # @是矩阵乘法
        # R t 都是map 到传感器坐标系的
        P = cp.asarray(K @ cp.concatenate([R, t[:, None]], 1), dtype=np.float32)
        # 作为一个
        # 传感器坐标系到map的t - self.center
        # 相当与一个偏执项 传感器到机器人中心的距离
        t_cam_map = -R.T @ t - self.center
        # 使用 .get() 获取结果，表明从 GPU 转移到 CPU。
        t_cam_map = t_cam_map.get()
        # 传感器位置在高程图 上的 索引
        x1 = cp.uint32((self.cell_n / 2) + ((t_cam_map[0]) / self.resolution))
        y1 = cp.uint32((self.cell_n / 2) + ((t_cam_map[1]) / self.resolution))
        # 传感器位置相对于 机器人位置高度 的差
        z1 = cp.float32(t_cam_map[2])

        self.uv_correspondence *= 0
        self.valid_correspondence[:, :] = False
        # self.distance_correspondence *= 0.0

        with self.map_lock:
            # 遍历的是整个高度图
            #  得到 高程图上的点 对应在图像上的坐标值,如果光线投射不通过,即直线上有高出的值 就认为无效
            # TODO 只投影了一个点,但是高程图栅格是一个大的范围
            self.image_to_map_correspondence_kernel(
                # 输入
                self.elevation_map,
                x1,
                y1,
                z1,
                # 变成一维
                P.reshape(-1),
                image_height,
                image_width,
                # 中心点
                self.center,
                # 输出
                # 高程图上的点 所对应的图像的坐标 及其是否有效
                self.uv_correspondence,
                self.valid_correspondence,
                size=int(self.cell_n * self.cell_n),
            )
            # # 更新语义层
            # self.semantic_map.update_layers_image(
            #     image, channels, self.uv_correspondence, self.valid_correspondence, image_height, image_width,
            # )
            self.new_semantic_map *= 0.0
            sem_map_idx = 0
            # self.color_correspondences_to_map_kernel(
            #     # 前一帧的语义图
            #     self.semantic_map,
            #     cp.uint64(sem_map_idx),
            #     # 彩色图像
            #     image,
            #     self.uv_correspondence,
            #     self.valid_correspondence,
            #     image_height,
            #     image_width,
            #     # 新得到的语义图
            #     self.new_semantic_map,
            #     size=int(self.cell_n * self.cell_n),
            # )
            self.gray_correspondences_to_map_kernel(
                # 前一帧的语义图
                self.semantic_map,
                cp.uint64(sem_map_idx),
                # 彩色图像
                image,
                self.uv_correspondence,
                self.valid_correspondence,
                image_height,
                image_width,
                # 新得到的语义图
                self.new_semantic_map,
                size=int(self.cell_n * self.cell_n),
            )
            self.semantic_map[sem_map_idx] = self.new_semantic_map[sem_map_idx]



def pub_elevationmap(odom_data_x,odom_data_y,height_data_local):

    grid_map = GridMap()
    # print(grid_map)
    grid_map.basic_layers.append('elevation')
    grid_map.layers.append('elevation')
    # grid_map.basic_layers.append('tra')

    grid_map.info.header.seq = 0
    grid_map.info.header.frame_id = "map"
    # grid_map.info.header.frame_id = "odom"
    grid_map.info.header.stamp = rospy.Time.now()

    # 设置地图分辨率、宽度、高度和原点坐标
    grid_map.info.resolution = 0.04
    grid_map.info.length_x = height_data_local.shape[0] * grid_map.info.resolution
    grid_map.info.length_y = height_data_local.shape[1] * grid_map.info.resolution
    # grid_map.info.pose.position.x = odom_data_x
    # grid_map.info.pose.position.y = odom_data_y
    grid_map.info.pose.position.x = 0
    grid_map.info.pose.position.y = 0
    grid_map.info.pose.position.z = 0
    grid_map.info.pose.orientation.x = 0
    grid_map.info.pose.orientation.y = 0
    grid_map.info.pose.orientation.z = 0
    grid_map.info.pose.orientation.w = 1

    mat = Float32MultiArray()
    mat.layout.dim.append(MultiArrayDimension())
    mat.layout.dim.append(MultiArrayDimension())
    # mat.layout.dim[0].label = "column_index"
    # mat.layout.dim[1].label = "row_index"
    mat.layout.dim[0].label = "column_index"
    mat.layout.dim[1].label = "row_index"
    mat.layout.dim[0].size = height_data_local.shape[0]
    mat.layout.dim[1].size = height_data_local.shape[1]
    # mat.layout.dim[0].stride = height_data_local.shape[1] * height_data_local.shape[0]
    # mat.layout.dim[1].stride = height_data_local.shape[1]
    mat.layout.data_offset = 0
    mat.data = height_data_local.flatten()

    grid_map.data.append(mat)

    return grid_map

# no use
def create_pointcloud_message(points):
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "lidar"  # Replace with your frame ID

    # Define the fields
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('semantic', 12, PointField.UINT8, 1)  # Assuming semantic info is in the last column
    ]

    # Create the PointCloud2 message
    pc_msg = pc2.create_cloud(header, fields, points)

    # Add semantics data
    pc_msg.data = np.array(points[:, 3], dtype=np.uint8).tobytes()

    return pc_msg



def project(points, image, M1, M2):
    """
    points: Nx3
    image: opencv img, 表示要投影的图像
    M1: 内参矩阵 K, 4*4
    M2: 外参矩阵， 4*4

    return: points 在像素坐标系下的坐标 N*4, 实际只用 N*2

    """
    resolution = image.shape


    # 指定过滤范围
    min_x, max_x = -15.0, 15.0
    min_y, max_y = -15.0, 15.0
    min_z, max_z = -1.0, 2.5

    # 进行过滤
    filtered_points = points[(points[:, 0] >= min_x) & (points[:, 0] <= max_x) &
                             (points[:, 1] >= min_y) & (points[:, 1] <= max_y) &
                             (points[:, 2] >= min_z) & (points[:, 2] <= max_z)]
    coords = filtered_points
    ones = np.ones(len(coords)).reshape(-1, 1)
    coords = np.concatenate([coords, ones], axis=1)
    transform = copy.deepcopy(M1 @ M2).reshape(4, 4)
    coords = coords @ transform.T
    coords = coords[np.where(coords[:, 2] > 0)]

    coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
    coords[:, 0] /= coords[:, 2]
    coords[:, 1] /= coords[:, 2]

    coords = coords[np.where(coords[:, 0] > 0)]
    coords = coords[np.where(coords[:, 0] < resolution[1])]
    coords = coords[np.where(coords[:, 1] > 0)]
    coords = coords[np.where(coords[:, 1] < resolution[0])]

    return coords


points_get_semantic_fun = points_get_semantic()
points_get_friction_fun = points_get_friction()
def project_semantic_point(raw_points , semantic_image, friction_map, M1, M2):
    P_lidar = M1 @ M2
    # Update elevation map using point cloud input.
    raw_points = cp.asarray(raw_points, dtype= np.float32)
    # 过滤掉 nan
    raw_points = raw_points[~cp.isnan(raw_points).any(axis=1)]

    semantic_image = cp.asarray(semantic_image, dtype=np.float32)

    friction_map = cp.asarray(friction_map, dtype=np.float32)

    P_lidar = cp.asarray(P_lidar, dtype=np.float32)

    semantic_points = cp.full((raw_points.shape[0], 4), cp.nan, dtype=np.float32)
    # semantic_points = cp.full((raw_points.shape[0], 4), 1, dtype=np.float32)

    # semantic_points[:,0:3] = raw_points

    points_get_semantic_fun(raw_points,
                                              semantic_image,
                                              semantic_image.shape[0],
                                              semantic_image.shape[1],
                                              P_lidar,

                                              semantic_points,
                                              size=(raw_points.shape[0]),)

    valid_mask = ~cp.isnan(semantic_points).any(axis=1)
    
    if cp.sum(valid_mask) > 0:
        valid_points = semantic_points[valid_mask]
        valid_raw_points = raw_points[valid_mask]
        
        # 为有效点创建包含摩擦系数的输出
        result = cp.full((valid_points.shape[0], 5), cp.nan, dtype=np.float32)
        result[:, :4] = valid_points
        # print(f"有效投影点数量: {valid_points.shape[0]}")
        # print(f"语义值范围: {valid_points[:, 3].min()} - {valid_points[:, 3].max()}")
        
        # 获取摩擦系数
        friction_points = cp.full((valid_points.shape[0], 4), cp.nan, dtype=np.float32)
        points_get_friction_fun(valid_raw_points,
                             friction_map,
                             friction_map.shape[0],
                             friction_map.shape[1],
                             P_lidar,
                             friction_points,
                             size=(valid_points.shape[0]),)
        
        result[:, 4] = friction_points[:, 3]
        # print(f"摩擦系数范围: {friction_points[:, 3].min()} - {friction_points[:, 3].max()}")
        # print(f"非零摩擦系数点数: {cp.sum(friction_points[:, 3] > 0)}")

        result = cp.asnumpy(result)
        return result
    else:
        print("没有有效的投影点!")
        return np.zeros((0, 5), dtype=np.float32)

# nouse
import ros_numpy
def publish_pointcloud(semantic_points, header=0):
    pcl = np.zeros(semantic_points[:,0].shape)
    pcl["x"] = semantic_points[:,0]
    pcl["y"] = semantic_points[:, 1]
    pcl["z"] = semantic_points[:, 2]
    pcl["semantic"] = semantic_points[:, 3]
    pc2 = ros_numpy.msgify(PointCloud2, pcl)
    pc2.header = header

    return pc2



def show_with_opencv(image, coords=None):
    """
    image: opencv image
    coords: 像素坐标系下的点, N*4
    """
    canvas = image.copy()

    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    # 画点
    if coords is not None:
        for index in range(coords.shape[0]):
            p = (int(coords[index, 0]), int(coords[index, 1]))
            cv2.circle(canvas, p, 2, color=[0, 0, 255], thickness=1)
    canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    return canvas


def normal_img(img):
    img1 = 1-((img - np.min(img)) / (np.max(img) - np.min(img)))
    img1 = 255 * img1
    img1 = img1.astype(np.uint8)

    return img1

def normal_img_semantic(img):
    img1 = (img - 0) / (15 - 0)
    img1 = 255 * img1
    img1 = img1.astype(np.uint8)

    return img1

def normal_one_img_semantic(img):
    img1 = (img - 0) / (15 - 0)

    return img1

def normal_one_img(img):
    img1 = 1-((img - np.min(img)) / (np.max(img) - np.min(img)))
    return img1

def intrinsic_to_homogeneous(intrinsic_matrix):
    # 创建一个4x4的单位矩阵
    homogeneous_matrix = np.eye(4)

    # 将3x3的内参矩阵复制到左上角的3x3部分
    homogeneous_matrix[:3, :3] = intrinsic_matrix

    return homogeneous_matrix

def combine_rotation_translation(rotation_matrix, translation_vector):
    # 创建一个4x4的齐次矩阵
    homogeneous_matrix = np.eye(4)

    # 将3x3的旋转矩阵复制到左上角的3x3部分
    homogeneous_matrix[:3, :3] = rotation_matrix

    # 将3x1的位置向量复制到最右侧的一列
    homogeneous_matrix[:3, 3] = translation_vector

    return homogeneous_matrix



camera_topic = '/zed2i/zed_node/left/image_rect_color'
point_cloud2_topic = '/rslidar_points'
odom_topic = '/fixposition/odometry_enu'


from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry
rospy.init_node('ht_sem_ele')
camera_sub = rospy.Subscriber(camera_topic, Image, queue_size=10)
point_cloud_sub = rospy.Subscriber(point_cloud2_topic, PointCloud2, queue_size=10)
odom_sub = rospy.Subscriber(odom_topic, Odometry, queue_size=10)

bridge = CvBridge()


# 可视化open3D 还是 matplotlib(好像不矛盾)
visual_point = True
visual_semantic_point = False
visual_position = False
ROS_visual = False
detect_seg_img = True

if visual_point or visual_semantic_point:

    import open3d as o3d

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=1000, height=1000)

    to_reset=True
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name='pcd', width=1000, height=1000)
    # 可视化参数设置
    opt = vis.get_render_option()
    # 设置背景色
    opt.background_color = np.asarray([0, 0, 0])
    # 设置点云大小
    opt.point_size = 1
    opt.show_coordinate_frame = True

    pcd = o3d.geometry.PointCloud()


    vis.add_geometry(pcd)
    # 创建坐标轴的三维模型
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)

    # 将坐标轴添加到场景中
    vis.add_geometry(coordinate_frame)

    # 创建一个XY平面
    plane_size = 20.0  # 平面大小
    plane_height = -0.35 #平面高度
    plane_color = [0, 0, 0]  # 黑色

    # 创建平面的顶点和法线
    plane_points = np.array([[-plane_size/2, -plane_size/2, plane_height],
                             [plane_size/2, -plane_size/2, plane_height],
                             [plane_size/2, plane_size/2, plane_height],
                             [-plane_size/2, plane_size/2, plane_height]])
    plane_triangles = np.array([[0, 1, 2], [0, 2, 3]])
    plane_colors = [plane_color for _ in range(len(plane_triangles))]

    # 创建Open3D几何对象
    plane_mesh = o3d.geometry.TriangleMesh()
    # plane_mesh.vertices = o3d.utility.Vector3dVector(plane_points)
    plane_mesh.triangles = o3d.utility.Vector3iVector(plane_triangles)
    # plane_mesh.triangle_colors = o3d.utility.Vector3dVector(plane_colors)
    # 将平面添加到渲染窗口
    # 创建一个网格颜色
    # mesh_color = o3d.geometry.TriangleMesh.create_box().paint_uniform_color(plane_color)

    # 将平面添加到渲染窗口
    vis.add_geometry(plane_mesh, reset_bounding_box=False)
    # vis.add_geometry(mesh_color)

if visual_position :
    # matplotlib画图程序
    num_subplots = 3  # Change this to the number of subplots you want

    # Create a figure with multiple subplots
    fig, axs = plt.subplots(num_subplots, 1, figsize=(8, 6 * num_subplots))

if ROS_visual:
    print('start_ros')
    # ros节点
    rospy.init_node('ht_MTraMap_local_publisher', anonymous=True)
    pub = rospy.Publisher('/MTraMap_local', GridMap, queue_size=10)
    pub_semantic_pc = rospy.Publisher('/Semantic_pointcloud', PointCloud2, queue_size=10)
    # rate = rospy.Rate(5)  # 1 Hz

# 高程图
param = Parameter(
    use_chainer=False, weight_file="./weights.dat", plugin_config_file="./plugin_config.yaml",max_drift = 100000,enable_visibility_cleanup= True
)
elevation = ElevationMap(param)
layers = ["elevation", "variance", "traversability", "min_filter", "smooth", "inpaint"]
data = np.zeros((elevation.cell_n - 2, elevation.cell_n - 2), dtype=np.float32)
data_dynamic = np.zeros((elevation.cell_n - 2, elevation.cell_n - 2), dtype=np.float32)
data_uv_correspondence_u = np.zeros((elevation.cell_n - 2, elevation.cell_n - 2), dtype=np.float32)
data_uv_correspondence_v = np.zeros((elevation.cell_n - 2, elevation.cell_n - 2), dtype=np.float32)
data_elevation_ls = np.zeros((elevation.cell_n - 2, elevation.cell_n - 2), dtype=np.float32)
data_semantic_map = np.zeros((elevation.cell_n - 2, elevation.cell_n - 2), dtype=np.float32)
updatePoseFps = 100
updateVarianceFps = 5


duration_updatePoseFps = 1.0 / (updatePoseFps + 0.00001)
duration_updateVarianceFps = 1.0 / (updateVarianceFps + 0.00001)
duration_timeInterval = 0.1
duration_visual_plot_Interval = 0.1

R_map_from_baselink_visual = np.identity(3)
R_map_from_baselink= np.identity(3)
R_map_from_rslidarm1link= np.identity(3)
R_map_from_zed= np.identity(3)
R_zed_from_map= np.identity(3)
t_vector = np.zeros(3)
t_map_from_baselink= np.zeros(3)
t_map_from_rslidarm1link= np.zeros(3)
t_map_from_zed= np.zeros(3)
t_zed_from_map= np.zeros(3)

R_t_read = False
t_init = 0
t_now = 0
t_last_poseupdate = 0
t_last_varianceupdate = 0
t_last_timeupdate = 0

t_last_visual_plot = 0
# print(t_vector)

t_vector_z_list = []
time_list = []

points = np.ones((1,3))


Tr = [
    -0.004017343, -0.9999754686, -0.005737875, 0.0540331795,
    -0.007702066, 0.0057686925, -0.999953699, -0.2201291155,
    0.9999622689, -0.0039729637, -0.007725052, -0.0907319794,
    0.0, 0.0, 0.0, 1.0
]


T_zed_from_rslidarm1link_1 = np.array(Tr).reshape(4, 4).astype(np.float32)

# Scout

K_zed = np.array([[267.158, 0.0, 311.137], [0.0, 267.158, 176.519], [0.0, 0.0, 1.0]])
T_K_zed = intrinsic_to_homogeneous(K_zed)

# 保存检测效果
method_name = 'elemap_raw_without_maxdrift'
raw_path = os.path.join('/home/lyx/ht_sem_elemap/ht_elevation_map_research_result',method_name)
creat_dir(raw_path)
# 初始化视频编写器
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_elemap = cv2.VideoWriter(os.path.join(raw_path,'elemap.mp4'), fourcc, 15.0,
                      (elevation.cell_n-2, elevation.cell_n-2))
out_ht_semantic_map_numpy1 = cv2.VideoWriter(os.path.join(raw_path,'ht_semantic_map_numpy1.mp4'), fourcc, 15.0,
                      (elevation.cell_n-2, elevation.cell_n-2))
out_ht_semantic_map_dynamic_numpy1 = cv2.VideoWriter(os.path.join(raw_path,'ht_semantic_map_dynamic_numpy1.mp4'), fourcc, 15.0,
                      (elevation.cell_n-2, elevation.cell_n-2))
out_results_seg1 = cv2.VideoWriter(os.path.join(raw_path,'results_seg1.mp4'), fourcc, 15.0,
                      (640, 360))
out_canvas = cv2.VideoWriter(os.path.join(raw_path,'canvas.mp4'), fourcc, 15.0,
                      (640, 360))
out_friction_static = cv2.VideoWriter(os.path.join(raw_path, 'friction_static.mp4'), fourcc, 15.0,
                      (elevation.cell_n-2, elevation.cell_n-2))
out_friction_dynamic = cv2.VideoWriter(os.path.join(raw_path, 'friction_dynamic.mp4'), fourcc, 15.0,
                      (elevation.cell_n-2, elevation.cell_n-2))

# TODO  这里更换语义分割模型
# 下面里面的路径要改一下
# ht_seg = ht_bisenet()
# ht_seg = lyx_deeplabv3plus()
ht_seg = lyx_deeplabv3plus_fri()


while not rospy.is_shutdown():

    t= rospy.get_time()

    if t_init == 0:
        t_init = t
    t_now = t - t_init


    if (t_now-t_last_varianceupdate) >= duration_updateVarianceFps:
        elevation.update_variance()
        t_last_varianceupdate = t_now
    if (t_now-t_last_timeupdate) >= duration_timeInterval:
        elevation.update_time()
        t_last_timeupdate = t_now

    camera_msg = rospy.wait_for_message(camera_topic, Image)

    t_camera_start = time.time()
    cv_image1 = bridge.imgmsg_to_cv2(camera_msg, "bgr8")
    alpha = 1.0  # 对比度保持不变
    beta = 0.2 * 255  # 亮度提升20%（相当于1.2倍）
    cv_image = cv2.convertScaleAbs(cv_image1, alpha=alpha, beta=beta)

    # 点云到图像坐标系的转换

    project_coords = project(points, cv_image, T_K_zed, T_zed_from_rslidarm1link_1)
    cv2.imshow('img',cv_image)
    results_seg_virtual, results_seg, friction_map = ht_seg(cv_image)
    results_seg1 = normal_img_semantic(results_seg)
    results_seg1 = cv2.applyColorMap(results_seg1, cv2.COLORMAP_JET)


    project_semantic_point_t1 = time.time()
    # 得到语义点云
    # semantic_points已经变成(N, 5)了，N为有效投影点数量，前3个通道为x,y,z, 第4个通道为语义值，第5个通道为摩擦系数
    semantic_points = project_semantic_point(points, results_seg, friction_map, T_K_zed, T_zed_from_rslidarm1link_1)
    project_semantic_point_t2 = time.time()
    # print('project_semantic_point_time:',project_semantic_point_t2-project_semantic_point_t1)

    # 输入语义点云
    elevation.input_semantic_points(semantic_points, R_map_from_rslidarm1link, t_map_from_rslidarm1link, 0, 0)
    # 添加GPU同步确保核函数执行完成
    cp.cuda.runtime.deviceSynchronize()
    
    # 检查CUDA错误
    try:
        # CuPy的正确错误检查方法
        cp.cuda.Device().synchronize()
    except Exception as e:
        print(f"CUDA同步错误: {e}")
        continue


    # 图层可视化
    ht_semantic_map_numpy = cp.asnumpy(elevation.ht_semantic_map_count[0,:,:])
    ht_semantic_map_mask_numpy = cp.asnumpy(elevation.ht_semantic_map_count[1, :, :])
    ht_semantic_map_mask_numpy =ht_semantic_map_mask_numpy.astype(np.uint8)
    ht_semantic_map_numpy = normal_img_semantic(ht_semantic_map_numpy)
    ht_semantic_map_numpy  = cv2.applyColorMap(ht_semantic_map_numpy , cv2.COLORMAP_JET)
    ht_semantic_map_numpy[ht_semantic_map_mask_numpy == 0] = [255,255,255]
    ht_semantic_map_numpy1 = cv2.flip(ht_semantic_map_numpy ,0)
    ht_semantic_map_numpy1 = cv2.flip(ht_semantic_map_numpy1, 1)
    cv2.imshow('ht_semantic_map_numpy', ht_semantic_map_numpy1)
    out_ht_semantic_map_numpy1.write(ht_semantic_map_numpy1)

    try:
        # 摩擦系数地图处理
        elevation.lyx_friction_map_count[0, :, :] = elevation.lyx_friction_map[0, :, :] / (elevation.lyx_friction_map[1, :, :] + 0.0001)
        elevation.lyx_friction_map_count[1, :, :] = elevation.lyx_friction_map_dynamic[0, :, :] / (elevation.lyx_friction_map_dynamic[1, :, :] + 0.0001)
    except Exception as e:
        print(f"摩擦地图计算错误: {e}")
        # 重新初始化摩擦地图
        elevation.lyx_friction_map_count = cp.zeros((2, elevation.cell_n, elevation.cell_n), dtype=elevation.data_type)

    # 获取摩擦系数地图数据
    friction_static_numpy = cp.asnumpy(elevation.lyx_friction_map_count[0, :, :])
    friction_mask_static = cp.asnumpy(elevation.lyx_friction_map[1, :, :] > 0)
    friction_mask_static = friction_mask_static.astype(np.uint8)

    # 归一化处理 - 摩擦系数通常在0-1之间
    # 可以根据实际摩擦系数范围调整
    friction_static_numpy = np.clip(friction_static_numpy, 0, 1)
    friction_static_vis = (friction_static_numpy * 255).astype(np.uint8)
    friction_static_vis = cv2.applyColorMap(friction_static_vis, cv2.COLORMAP_JET)  # 低摩擦(蓝)到高摩擦(红)
    friction_static_vis[friction_mask_static == 0] = [255, 255, 255]  # 无数据区域设为白色

    # 翻转处理以匹配显示方向
    friction_static_vis = cv2.flip(friction_static_vis, 0)
    friction_static_vis = cv2.flip(friction_static_vis, 1)

    # 显示并保存
    cv2.imshow('lyx_friction_map_static', friction_static_vis)
    out_friction_static.write(friction_static_vis)

    # # 摩擦系数地图可视化 - 动态物体
    # friction_dynamic_numpy = cp.asnumpy(elevation.lyx_friction_map_count[1, :, :])
    # friction_mask_dynamic = cp.asnumpy(elevation.lyx_friction_map_dynamic[1, :, :] > 0)
    # friction_mask_dynamic = friction_mask_dynamic.astype(np.uint8)

    # # 归一化处理
    # friction_dynamic_numpy = np.clip(friction_dynamic_numpy, 0, 1)  
    # friction_dynamic_vis = (friction_dynamic_numpy * 255).astype(np.uint8)
    # friction_dynamic_vis = cv2.applyColorMap(friction_dynamic_vis, cv2.COLORMAP_JET)
    # friction_dynamic_vis[friction_mask_dynamic == 0] = [255, 255, 255]  # 无数据区域设为白色

    # # 翻转处理以匹配显示方向
    # friction_dynamic_vis = cv2.flip(friction_dynamic_vis, 0)
    # friction_dynamic_vis = cv2.flip(friction_dynamic_vis, 1)

    # # 显示并保存
    # cv2.imshow('Friction Dynamic', friction_dynamic_vis)
    # out_friction_dynamic.write(friction_dynamic_vis)

    if ROS_visual:
        ros_semantic_points =publish_pointcloud(semantic_points)
        pub_semantic_pc.publish(ros_semantic_points)
    if visual_semantic_point :

        pcd.points = o3d.utility.Vector3dVector(semantic_points[:, :3])  # 提取前三列作为点的坐标

        import matplotlib.cm as cm
        # 将单一值映射到 RGB 颜色
        cmap = cm.get_cmap('jet')  # 选择颜色映射，这里使用 viridis

        semantic_color = cmap(semantic_points[:, 3])[:, :3]


        pcd.colors = o3d.utility.Vector3dVector(semantic_color)  # 使用语义信息作为点的颜色



        # 设置颜色
        pcd.colors = o3d.utility.Vector3dVector(semantic_color)

        vis.update_geometry(pcd)
        if to_reset:
            vis.reset_view_point(True)
            to_reset = False

        vis.poll_events()
        vis.update_renderer()



    cv2.imshow('results_seg',results_seg1)
    out_results_seg1.write(results_seg1)
    t_camera_end = time.time()



    if visual_position and (t_now-t_last_visual_plot) >= duration_visual_plot_Interval and len(time_list)>0:
        t_last_visual_plot = t_now
        for ax in axs:
            ax.cla()

        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        for ax in axs:
            ax.plot(time_list, t_vector_z_list, "-g")
            ax.set_ylim(t_vector_z_list[-1]-0.2, t_vector_z_list[-1]+0.2)
            ax.set_xlim(time_list[-1]-5, time_list[-1])
            # ax.axis("equal")

        plt.pause(0.000000001)

    point_cloud_msg = rospy.wait_for_message(point_cloud2_topic, PointCloud2)

    lidar = pc2.read_points(point_cloud_msg,field_names=("x", "y", "z"))

    points = np.array(list(lidar))

    elevation.get_map_with_name_ref("elevation", data)
    elemap_data = data.copy()
    nan_mask = np.isnan(elemap_data)

    elemap_data[nan_mask] = 0


    elemap_data1 = 1-(elemap_data - np.min(elemap_data)) / (np.max(elemap_data) - np.min(elemap_data))

    elemap_data1 = 255 *elemap_data1
    elemap_data1 = elemap_data1.astype(np.uint8)

    elemap_data1 = cv2.applyColorMap(elemap_data1, cv2.COLORMAP_RAINBOW)
    elemap_data1[nan_mask] = [255,255,255]

    cv2.imshow('elemap',elemap_data1)
    out_elemap.write(elemap_data1)



    cv2.waitKey(1)

    points_maprpy = (np.dot(R_map_from_baselink_visual, points.T)).T

    if visual_point:
        # 设置相机位置和视点
        view_control = vis.get_view_control()

        view_control.set_lookat([0, 0, 2])  # 设置视点
        view_control.set_front((0, 0, 1)) # set the positive direction of the x-axis toward you
        view_control.set_up((1, 0, 0))  # set the positive direction of the x-axis as the up direction


        pcd.points = o3d.utility.Vector3dVector(points_maprpy)
        pcd.paint_uniform_color([1, 0, 0])


        vis.update_geometry(pcd)
        if to_reset:
            vis.reset_view_point(True)
            to_reset = False

        vis.poll_events()
        vis.update_renderer()


        odom_msg = rospy.wait_for_message(odom_topic, Odometry)
        orientation_euler = tf.transformations.euler_from_quaternion([
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w
        ], axes='sxyz')

        R_matrix_visual = tf.transformations.euler_matrix(orientation_euler[0],orientation_euler[1],0, 'sxyz')[:3,:3]

        R_matrix = tf.transformations.euler_matrix(orientation_euler[0], orientation_euler[1], orientation_euler[2], 'sxyz')[:3, :3]

        R_map_from_rslidarm1link = R_matrix
        t_map_from_rslidarm1link = t_vector


        t_vector = [odom_msg.pose.pose.position.x,odom_msg.pose.pose.position.y,odom_msg.pose.pose.position.z]
        # t_vector = [0,0,0]
        R_t_read = True
        if (t_now-t_last_poseupdate) >= duration_updatePoseFps:
            elevation.move_to(t_vector)
            t_last_poseupdate = t_now


        t_vector_z_list = np.append(t_vector_z_list, t_vector[2])
        time_list = np.append(time_list, t_now)

out_elemap.release()
out_ht_semantic_map_numpy1.release()
out_ht_semantic_map_dynamic_numpy1.release()
out_results_seg1.release()
out_canvas.release()
out_friction_static.release()
out_friction_dynamic.release()




