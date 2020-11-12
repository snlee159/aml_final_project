from project.local_config import LocalConfig

class GlobalConfig:

    # LocalConfig variables
    RAW_VIDEO_DIR_PATH = LocalConfig.RAW_VIDEO_DIR_PATH
    CLIPPED_VIDEO_DIR_PATH = LocalConfig.CLIPPED_VIDEO_DIR_PATH
    FRAMES_BASE_PATH = LocalConfig.FRAMES_BASE_PATH
    CLIPPED_VIDEO_DIR_CLEANED_PATH = LocalConfig.CLIPPED_VIDEO_DIR_CLEANED_PATH
    FRAMES_CLEANED_BASE_PATH = LocalConfig.FRAMES_CLEANED_BASE_PATH

    # vgg16 cat labels
    VGG16_LABELS = ['tabby',
                    'tiger_cat',
                    'Persian_cat',
                    'Siamese_cat',
                    'Egyptian_cat']
    VGG16_CAT_LABEL_INDICES = [281, 282, 283, 284, 285]

    # feature names
    PIXEL_CHANGES = 'Pixel_changes'

    # Regression types
    LINEAR_REG_TYPE = 'Linear'
    LASSO_REG_TYPE = 'Lasso'
    RIDGE_REG_TYPE = 'Ridge'