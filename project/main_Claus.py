import pickle
import numpy as np
from project.global_config import GlobalConfig
from project.cat_popularity import cat_popularity_classification
from project.cat_video_editing import limit_video_length, extract_frames
from project.vgg16_cat_detector import load_vgg16_model
from project.vgg16_cat_detector import preprocess_and_predict_frames
from project.feature_extraction import load_feature_extraction_model
from project.feature_extraction import extract_pixel_change_feature
from project.feature_extraction import extract_viewcount_feature_from_raw_videos
from project.feature_extraction import extract_video_title_from_raw_videos
from project.cat_video_editing import clean_dataset_with_dict
from project.feature_visualizations import plot_pixel_change_feature_histogram
from project.feature_visualizations import plot_view_count_feature_histogram
from project.feature_extraction import make_view_count_feature_categorical
from Regression.data_preprocessing import extract_regression_features
from Regression.regression_model import train_regression_model
from project.feature_extraction import hog_feature_extraction


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
# vgg16_model = load_vgg16_model()
# cat_video_boolean_dict = preprocess_and_predict_frames(model=vgg16_model,
#                                                        frame_dir=GlobalConfig.FRAMES_BASE_PATH)
# print(cat_video_boolean_dict)
#
# clean_dataset_with_dict(cat_video_boolean_dict='cat_video_boolean_dict.pkl')


# ===== feature extraction =====
# feature_extraction_model = load_feature_extraction_model()
#extract_pixel_change_feature(frame_dir=GlobalConfig.FRAMES_CLEANED_BASE_PATH,
#                             make_dummy_feature=True, threshold=75)
#extract_viewcount_feature_from_raw_videos('cat_video_boolean_dict.pkl')
#make_view_count_feature_categorical('../pkl/view_count_feature_dict.pkl')
# extract_video_title_from_raw_videos('cat_video_boolean_dict.pkl')
for i in range(2, 4):
    hog_feature_extraction(num_clusters=i)

# ===== plot feature plots =====
#plot_pixel_change_feature_histogram('../pkl/pixel_change_feature_dict.pkl', n_bins=30)
#plot_view_count_feature_histogram('../pkl/view_count_feature_dict.pkl', n_bins=5, count_limit=50000)


# ===== Regression Model =====
#X_train, y_train, X_test, y_test = extract_regression_features(sound_feature_list=['bpm', 'beat_ratio'],#['bpm', 'beat_ratio', 'zcr_mean','energy_mean','energy_entropy_mean', 'spectral_centroid_mean', 'spectral_spread_mean', 'spectral_entropy_mean', 'spectral_flux_mean','spectral_rolloff_mean', 'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean', 'mfcc_4_mean', 'mfcc_5_mean', 'mfcc_6_mean', 'mfcc_7_mean'],
#                                                               pixel_change=True, pixel_change_dummy=False,
#                                                               reduce_pixel_change_to_one_variable=False,
#                                                               labels_categorical=False,
#                                                               train_test_split_ratio=0.8)

#train_regression_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
#                       type=GlobalConfig.LINEAR_REG_TYPE)
#train_regression_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
#                       type=GlobalConfig.LASSO_REG_TYPE)
#train_regression_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
#                       type=GlobalConfig.RIDGE_REG_TYPE)


#with open('../pkl/hog_feature_dict_K_2.pkl', 'rb') as file:
#    dic = pickle.load(file)

#print()
#print('Success at running')






