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
NUM_CLASSES = 18
IMG_RESIZE_DIM = 512                                                        # Image dimension (in pixels) for training and inference (512 in paper)
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

METAINFO = {
    "classes": (
        "asphalt",
        "dirt",
        "mud",
        "water",
        "gravel",
        "other-terrain",
        "tree-trunk",
        "tree-foliage",
        "bush",
        "fence",
        "structure",
        "pole",
        "vehicle",
        "rock",
        "log",
        "other-object",
        "sky",
        "grass",
    ),
    "palette": [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (230, 25, 75),
        (0, 128, 128),
        (170, 110, 40),
        (255, 250, 200),
        (128, 0, 0),
        (170, 255, 195),
        (128, 128, 0),
        (250, 190, 190),
        (0, 0, 128),
        (128, 128, 128),
    ],
        "cidx": [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17
        ]
    }