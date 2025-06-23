import numpy as np
import yaml
import scipy.stats as stats
import cv2

class GaussianFrictionModel:
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance

    def get_parameters(self):
        return self.mean, self.variance

class FrictionCalculator:
    def __init__(self, yaml_path="/home/lyx/ht_sem_elemap/deeplabv3plus/WildScenes_categories.yaml", num_classes=19):
        """初始化摩擦系数计算器"""
        self.num_classes = num_classes
        self.ignore_indices = [3, 5, 10, 12, 15, 16]  # 忽略的类别索引
        self.terrain_classes = self.load_terrain_properties(yaml_path)
        self.friction_models = self.initialize_friction_models()
    
    def load_terrain_properties(self, yaml_path):
        """从YAML文件加载地形属性"""
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        return data
    
    def initialize_friction_models(self):
        """初始化摩擦系数模型"""
        friction_models = [None] * self.num_classes
        default_model = GaussianFrictionModel(mean=0.01, variance=0.99)
        
        # 遍历YAML中定义的每个类别
        for class_name, properties in self.terrain_classes.items():
            index = properties.get('index', None)
            if index is None or index >= self.num_classes:
                continue
                
            # 获取摩擦系数和方差
            friction_info = properties.get('friction', {})
            mean = friction_info.get('mean', None)
            variance = friction_info.get('variance', None)
            
            if mean is not None and variance is not None:
                friction_models[index] = GaussianFrictionModel(mean, variance)
            else:
                friction_models[index] = default_model
        
        # 替换未定义的类别为默认模型
        for i in range(self.num_classes):
            if friction_models[i] is None:
                friction_models[i] = default_model
                
        return friction_models
    
    def assign_friction(self, class_probs):
        """
        基于类别概率分配摩擦系数
        
        参数:
            class_probs: shape为[num_classes, height, width]的数组，表示每个像素的类别概率分布
            
        返回:
            friction_map: 摩擦系数图 [height, width]
            friction_variance: 摩擦系数方差 [height, width]
            friction_color: 摩擦系数热力图 [height, width, 3]
        """
        friction_map = np.zeros(class_probs.shape[1:], dtype=np.float32)
        E_f_squared = np.zeros(class_probs.shape[1:], dtype=np.float32)
        
        # 计算每个像素的摩擦系数均值和均方值
        for i in range(self.num_classes):
            mean, variance = self.friction_models[i].get_parameters()
            friction_map += class_probs[i] * mean  # 计算 E[f]
            E_f_squared += class_probs[i] * (mean**2 + variance)  # 计算 E[f^2]
        
        # 计算摩擦系数方差: Var[f] = E[f^2] - (E[f])^2
        friction_variance = E_f_squared - friction_map**2
        friction_variance[friction_variance < 0] = 0.01  # 处理负的方差
        
        # 获取预测的类别
        predicted_classes = np.argmax(class_probs, axis=0)
        
        # 处理忽略的区域
        ignore_mask = np.isin(predicted_classes, self.ignore_indices)
        friction_map[ignore_mask] = 0.01
        friction_variance[ignore_mask] = 0.99
        
        # 生成热力图可视化
        friction_normalized = (friction_map - np.min(friction_map)) / (np.max(friction_map) - np.min(friction_map) + 1e-8)
        friction_normalized = np.clip(friction_normalized, 0, 1)
        friction_color = cv2.applyColorMap((friction_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        return friction_map, friction_variance, friction_color