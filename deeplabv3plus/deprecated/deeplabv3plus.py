import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict

class AtrousSeparableConvolution(nn.Module):
    """空洞可分离卷积
    将标准卷积分解为深度卷积和逐点卷积，减少参数量和计算复杂度
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                stride=1, padding=0, dilation=1, bias=False):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # 深度卷积 - 每个通道单独卷积
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                    stride=stride, padding=padding, dilation=dilation, 
                    bias=bias, groups=in_channels),
            # 逐点卷积 - 融合通道信息
            nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                    stride=1, padding=0, bias=bias),
        )
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    """ASPP的卷积分支"""
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, 
                    dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    """ASPP的全局池化分支"""
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    """空洞空间金字塔池化模块
    用于捕获多尺度上下文信息
    """
    def __init__(self, in_channels, atrous_rates=[6, 12, 18], out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        # 1x1 卷积分支
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        # 多个空洞卷积分支
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        
        # 全局池化分支
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 融合所有分支
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)  # 添加dropout防止过拟合
        )
        self._init_weight()

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabHeadV3Plus(nn.Module):
    """DeepLabV3+ 解码头
    结合低级特征和高级特征进行精细分割
    """
    def __init__(self, high_level_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        
        # 低级特征处理
        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # 高级特征处理
        self.aspp = ASPP(high_level_channels, aspp_dilate)

        # 特征融合和分类
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, features):
        low_level_feature = self.project(features['low_level'])
        output_feature = self.aspp(features['out'])
        
        # 上采样高级特征到低级特征的尺寸
        output_feature = F.interpolate(
            output_feature, 
            size=low_level_feature.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # 拼接特征并分类
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class Deeplabv3Plus(nn.Module):
    """DeepLabV3+ 语义分割模型"""
    def __init__(self, num_classes, backbone_name='resnet101', 
                output_layer_high='layer4', output_layer_low='layer1',
                aspp_dilate=[6, 12, 18]): 
        super(Deeplabv3Plus, self).__init__()
        
        # 存储需要的输出层
        self.output_layer_high = output_layer_high
        self.output_layer_low = output_layer_low

        # 加载预训练骨干网络
        self.backbone_name = backbone_name
        self.backbone, self.high_level_channels = self._get_backbone_and_channels(backbone_name)
        
        # 获取低级特征通道数
        low_level_layers = {'layer1': 256, 'layer2': 512, 'layer3': 1024}
        low_level_channels = low_level_layers[output_layer_low]
        
        # 解码头
        self.decoder = DeepLabHeadV3Plus(
            high_level_channels=self.high_level_channels,
            low_level_channels=low_level_channels,
            num_classes=num_classes,
            aspp_dilate=aspp_dilate
        )
        
        

    def _get_backbone_and_channels(self, backbone_name):
        """获取骨干网络和对应的通道数"""
        if backbone_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            out_channels = 2048 if self.output_layer_high == 'layer4' else 1024
        elif backbone_name == 'resnet101':
            model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
            out_channels = 2048 if self.output_layer_high == 'layer4' else 1024
        else:
            raise ValueError(f"不支持的骨干网络: {backbone_name}")
        return model, out_channels

    def forward(self, x):
        input_shape = x.shape[-2:]
        
        # 提取特征
        features = self._extract_features(x)
        
        # 解码并上采样到原始尺寸
        x = self.decoder(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        return x
    
    def _extract_features(self, x):
        """提取骨干网络特征"""
        features = OrderedDict()
        
        # ResNet的阶段划分
        stages = {
            'layer1': [0, 1, 2, 3, 4],
            'layer2': [0, 1, 2, 3, 4, 5],
            'layer3': [0, 1, 2, 3, 4, 5, 6],
            'layer4': [0, 1, 2, 3, 4, 5, 6, 7]
        }
        
        # 获取模型的各层
        children = list(self.backbone.children())
        
        # 获取低级特征
        for i in range(stages[self.output_layer_low][-1] + 1):
            x = children[i](x)
            if i == stages[self.output_layer_low][-1]:
                features['low_level'] = x
        
        # 获取高级特征
        for i in range(stages[self.output_layer_low][-1] + 1, stages[self.output_layer_high][-1] + 1):
            x = children[i](x)
            if i == stages[self.output_layer_high][-1]:
                features['out'] = x
                
        return features

def convert_to_separable_conv(module):
    """将模型中的标准卷积转换为可分离卷积"""
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
        new_module = AtrousSeparableConvolution(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.bias is not None
        )
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module