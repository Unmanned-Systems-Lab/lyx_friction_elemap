import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import cv2
import argparse

from deeplabv3plus.config import DEVICE, NUM_CLASSES, METAINFO, IMG_RESIZE_DIM, experiments
from deeplabv3plus.deeplabv3plus import Deeplabv3Plus

def load_model(model_path, config, device=DEVICE):
    """加载预训练的DeepLabV3+模型"""
    model = Deeplabv3Plus(num_classes=NUM_CLASSES, 
                          backbone_name=config["backbone"], 
                          output_layer_high=config["output_layer_high"], 
                          output_layer_low=config["output_layer_low"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path=None, rgb_image=None):
    """预处理输入RGB图像"""
    # 图像可以从路径加载或直接作为RGB数组传入
    if image_path is not None:
        # 从路径加载图像
        image = Image.open(image_path).convert("RGB")
    elif rgb_image is not None:
        # 使用提供的RGB图像（可以是numpy数组或PIL图像）
        if isinstance(rgb_image, np.ndarray):
            image = Image.fromarray(rgb_image.astype('uint8'))
        else:
            image = rgb_image
    else:
        raise ValueError("必须提供image_path或rgb_image之一")
    
    # 保存原始尺寸以便后处理
    original_size = image.size[::-1]  # (height, width)
    
    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize((IMG_RESIZE_DIM, IMG_RESIZE_DIM)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    return input_tensor, original_size, image

def segment_image(model, input_tensor, original_size, device=DEVICE):
    """使用模型进行图像分割"""
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        
    # 获取预测类别
    probs = torch.softmax(output, dim=1)
    predictions = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
    
    # 将预测结果调整回原始尺寸
    predictions_resized = cv2.resize(predictions.astype(np.uint8), 
                                    (original_size[1], original_size[0]), 
                                    interpolation=cv2.INTER_NEAREST)
    
    return predictions_resized

def visualize_segmentation(segmentation_map):
    """将分割图可视化为彩色图像"""
    # 创建彩色分割图
    colored_segmentation = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)
    
    # # 使用配置中的调色板为每个类别着色
    # for class_idx, color in enumerate(METAINFO['palette']):
    #     colored_segmentation[segmentation_map == class_idx] = color

    for class_idx, color in enumerate(METAINFO['palette']):
        # 使用cidx中定义的索引值
        cidx_value = METAINFO['cidx'][class_idx]
        if cidx_value != 255:  # 跳过未标记的类别
            colored_segmentation[segmentation_map == cidx_value] = color
    
    return colored_segmentation

def save_segmentation(colored_segmentation, output_path="segmentation_result.png"):
    """保存分割结果到文件"""
    result_image = Image.fromarray(colored_segmentation)
    result_image.save(output_path)
    return output_path

def infer_rgb_image(rgb_image=None, image_path=None, model_path="last_model_0.148.pth", 
                   output_path=None, config=None):
    """
    主函数：接收RGB图像或图像路径，返回分割结果
    
    参数:
        rgb_image: numpy数组或PIL图像，RGB格式的输入图像
        image_path: 字符串，输入图像的路径
        model_path: 字符串，预训练模型的路径
        output_path: 字符串，分割结果保存的路径（如果为None则不保存）
        config: 字典，模型配置参数
    
    返回:
        segmentation_map: numpy数组，分割类别图（每个像素值为类别索引）
        colored_segmentation: numpy数组，可视化的RGB分割图
    """
    # 默认配置
    if config is None:
        config = experiments[0]  # 使用第一个配置
    
    # 加载模型
    model = load_model(model_path, config)
    
    # 预处理图像
    input_tensor, original_size, _ = preprocess_image(image_path, rgb_image)
    
    # 进行分割
    segmentation_map = segment_image(model, input_tensor, original_size)
    
    # 可视化分割结果
    colored_segmentation = visualize_segmentation(segmentation_map)
    
    # 保存结果（如果需要）
    if output_path:
        save_segmentation(colored_segmentation, output_path)
    
    return segmentation_map, colored_segmentation

# 使用示例
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='用于单张图像分割的DeepLabV3+')
    parser.add_argument('--image', type=str, help='输入图像路径')
    parser.add_argument('--model', type=str, default="best_model_0.189.pth", help='模型路径')
    parser.add_argument('--output', type=str, default="segmentation_result_1.png", help='输出图像路径')
    args = parser.parse_args()
    
    # 从配置文件获取默认配置
    config = experiments[0]
    
    # 进行推理
    _, colored_segmentation = infer_rgb_image(
        image_path=args.image,
        model_path=args.model,
        output_path=args.output,
        config=config
    )
    
    print(f"分割结果已保存至 {args.output}")