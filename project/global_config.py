from project.local_config import LocalConfig

class GlobalConfig:

    # LocalConfig variables
    RAW_VIDEO_DIR_PATH = LocalConfig.RAW_VIDEO_DIR_PATH
    CLIPPED_VIDEO_DIR_PATH = LocalConfig.CLIPPED_VIDEO_DIR_PATH
    FRAMES_BASE_PATH = LocalConfig.FRAMES_BASE_PATH

    # vgg16 cat labels
    VGG16_LABELS = ['tabby',
                    'tiger_cat',
                    'Persian_cat',
                    'Siamese_cat',
                    'Egyptian_cat',
                    'polecat',
                    'Madagascar_cat']