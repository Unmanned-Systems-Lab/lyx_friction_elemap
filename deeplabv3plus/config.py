import torch

# DEVICE SELECTION
DEVICE = (
    "cuda"                                                                  # Prefers CUDA acceleration (NVIDIA) -> then Metal acceleration (MAC) -> then CPU
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# HYPERPARAMETERS
LEARNING_RATE = 3e-3
BATCH_SIZE = 8
NUM_EPOCHS = 100
NUM_WORKERS = 0                                                             # Sub-processes used by Data Loader (set to 0 for main thread only)
PIN_MEMORY = True
NUM_CLASSES = 15
IMG_RESIZE_DIM = 768                                                        # Image dimension (in pixels) for training and inference (512 in paper)
OUTPUT_DIM = (2016, 1512)                                                   # Output prediction dimension (in pixels) for 'IndexLabel' and 'label'

# PATHS
TRAIN_CSV = "splits/train.csv"                                              # to CSV files (for train-val-test-split to be used in 'CustomDataset' class)
VAL_CSV = "splits/val.csv"
TEST_CSV = "splits/test.csv"
WILDSCENES_PATH = "data/WildScenes"                                         # path to folder containing 'WildScenes2d'
PREDICTIONS = 'data/WildScenes/WildScenes2d/Test/predictions'               # location to save RGB predictions
PREDICTIONS_LABELS = 'data/WildScenes/WildScenes2d/Test/predictions_label'  # location to save 'indexLabel' style index predictions

# 阶段训练配置
FREEZE_EPOCHS = 50  # 冻结阶段的epoch数
UNFREEZE_EPOCHS = 50  # 解冻阶段的epoch数
TOTAL_EPOCHS = FREEZE_EPOCHS + UNFREEZE_EPOCHS

# 学习率配置
FREEZE_BATCH_SIZE = 8  # 冻结阶段批次大小
UNFREEZE_BATCH_SIZE = 4  # 解冻阶段批次大小
INIT_LR = 7e-3  # 初始学习率
MIN_LR = 7e-5  # 最小学习率
NBS = 16  # 标准化批次大小

# 模型保存配置
SAVE_PERIOD = 5  # 每隔多少个epoch保存模型
SAVE_DIR = './logs'


# Chosen experimental configurations
experiments = [
    # {
    #     "backbone": "resnet50",
    #     "output_layer_high": "layer4",
    #     "output_layer_low": "layer1"
    # },
    {
        "backbone": "resnet101",
        "output_layer_high": "layer4",
        "output_layer_low": "layer1"
    }
    # {
    #     "backbone": "resnet50",
    #     "output_layer_high": "layer3",
    #     "output_layer_low": "layer1"
    # },
    # {
    #     "backbone": "resnet101",
    #     "output_layer_high": "layer3",
    #     "output_layer_low": "layer1"
    # }
]

CLASSES_15 = [
    "bush", "dirt", "fence", "grass", "gravel", "log",
    "other-object", "mud", "structure", "other-terrain",
    "rock", "sky", "tree-foliage", "tree-trunk", "water"
]

# 定义15类的调色板 - 保留原始类别的颜色以便于比较
# 使用与dataset_v3.py中的LABEL_REMAP一致的映射关系
PALETTE_15 = [
    (230, 25, 75),   # bush (原bush)
    (60, 180, 75),   # dirt (原dirt)
    (0, 128, 128),   # fence (原fence)
    (128, 128, 128), # grass (原grass)
    (145, 30, 180),  # gravel (原gravel)
    (128, 128, 0),   # log (原log)
    (250, 190, 190), # other-object (原other-object/pole)
    (255, 225, 25),  # mud (原mud)
    (170, 110, 40),  # structure (原structure)
    (70, 240, 240),  # other-terrain (原other-terrain/asphalt)
    (170, 255, 195), # rock (原rock)
    (0, 0, 128),     # sky (原sky)
    (210, 245, 60),  # tree-foliage (原tree-foliage)
    (240, 50, 230),  # tree-trunk (原tree-trunk)
    (0, 130, 200),   # water (原water)
]

# 更新METAINFO为15类
METAINFO = {
    "classes": tuple(CLASSES_15),
    "palette": PALETTE_15,
    "cidx": list(range(NUM_CLASSES))
}