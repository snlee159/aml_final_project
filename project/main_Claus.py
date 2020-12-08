import pickle
from project.global_config import GlobalConfig
from project.cat_popularity import cat_popularity_classification
from project.cat_video_editing import limit_video_length, extract_frames
from project.vgg16_cat_detector import load_vgg16_model
from project.vgg16_cat_detector import preprocess_and_predict_frames
from project.feature_extraction import load_feature_extraction_model
from project.feature_extraction import extract_pixel_change_feature
from project.feature_extraction import extract_viewcount_feature_from_raw_videos
from project.feature_extraction import extract_video_title_from_raw_videos
from project.feature_extraction import hog_feature_extraction
from project.cat_video_editing import clean_dataset_with_dict
from Regression.data_preprocessing import extract_regression_features
from Regression.regression_model import train_test_regression_model


# ===== Download test video =====
# test = cat_popularity_classification('https://www.youtube.com/watch?v=jqjrfmSE4C0')


# ===== edit video =====
# limit_video_length(raw_video_data_path=GlobalConfig.RAW_VIDEO_DIR_PATH,
#                   start_time=10,
#                   end_time=40)

# ===== extract frames for classification purposes =====
# extract_frames(clipped_video_data_path=GlobalConfig.CLIPPED_VIDEO_DIR_PATH,
#                num_frames=15,
#                times=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29],
#                start_index=0, end_index=4000)


# ===== predict cat with vgg16 model =====
# vgg16_model = load_vgg16_model()
# cat_video_boolean_dict = preprocess_and_predict_frames(model=vgg16_model,
#                                                        frame_dir=GlobalConfig.FRAMES_BASE_PATH)
# print(cat_video_boolean_dict)
#
# clean_dataset_with_dict(cat_video_boolean_dict='../pkl_final/cat_video_boolean_dict.pkl')


# ===== feature extraction =====
# feature_extraction_model = load_feature_extraction_model()
# extract_pixel_change_feature(frame_dir=GlobalConfig.FRAMES_CLEANED_BASE_PATH)
# extract_viewcount_feature_from_raw_videos('cat_video_boolean_dict.pkl')
# extract_video_title_from_raw_videos('cat_video_boolean_dict.pkl')



# ===== Regression Model =====
# X_train, y_train, X_test, y_test = \
#     extract_regression_features(use_sound_features=True, sound_feature_list=['beat_ratio'],
#
#                                 use_HOG=False, K=20,
#
#                                 use_pixel_change=False,
#                                 actual_pixel_change_feature=False,
#                                 categorical_pixel_change_feature=False,
#                                 single_scene_detection=False,
#
#                                 use_title=True,
#
#                                 labels_categorical=True, train_test_split_ratio=0.8)
#
# train_test_regression_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
#                        type=GlobalConfig.LINEAR_REG_TYPE)
# train_test_regression_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
#                        type=GlobalConfig.LASSO_REG_TYPE)
# train_test_regression_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
#                        type=GlobalConfig.RIDGE_REG_TYPE)


#
with open('../pkl_final/all_view_count_dict.pkl', 'rb') as file:
    dic = pickle.load(file)

print()
#
# print()







