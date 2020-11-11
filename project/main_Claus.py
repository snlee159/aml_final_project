import pickle
from project.global_config import GlobalConfig
from project.cat_popularity import cat_popularity_classification
from project.cat_video_editing import limit_video_length, extract_frames
from project.vgg16_cat_detector import load_vgg16_model
from project.vgg16_cat_detector import preprocess_and_predict_frames
from project.feature_extraction import load_feature_extraction_model
from project.feature_extraction import extract_pixel_change_feature
from project.feature_extraction import extract_viewcount_feature_from_raw_videos
from project.cat_video_editing import clean_dataset_with_dict


# ===== Download test video =====
# test = cat_popularity_classification('https://www.youtube.com/watch?v=jqjrfmSE4C0')


# ===== edit video =====
# limit_video_length(raw_video_data_path=GlobalConfig.RAW_VIDEO_DIR_PATH,
#                   start_time=10,
#                   end_time=40)

# ===== extract frames for classification purposes =====
# extract_frames(clipped_video_data_path=GlobalConfig.CLIPPED_VIDEO_DIR_PATH,
#                num_frames=15,
#                times=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29])


# ===== predict cat with vgg16 model =====
vgg16_model = load_vgg16_model()
cat_video_boolean_dict = preprocess_and_predict_frames(model=vgg16_model,
                                                       frame_dir=GlobalConfig.FRAMES_BASE_PATH)
print(cat_video_boolean_dict)

clean_dataset_with_dict(cat_video_boolean_dict='cat_video_boolean_dict.pkl')


# ===== feature extraction =====
# feature_extraction_model = load_feature_extraction_model()
# extract_pixel_change_feature(frame_dir=GlobalConfig.FRAMES_CLEANED_BASE_PATH)
# extract_viewcount_feature_from_raw_videos('cat_video_boolean_dict.pkl')


print('Success at running')









