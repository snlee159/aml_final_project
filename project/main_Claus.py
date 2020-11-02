from project.global_config import GlobalConfig
from project.cat_popularity import cat_popularity_classification
from project.cat_video_editing import limit_video_length, extract_frames
from project.vgg16_cat_detector import load_vgg16_model
from project.vgg16_cat_detector import preprocess_and_predict_frames


# ===== Download video =====
#test = cat_popularity_classification('https://www.youtube.com/watch?v=Gb-4dtzIOO0')


# ===== edit video =====
#limit_video_length(raw_video_data_path=GlobalConfig.RAW_VIDEO_DIR_PATH,
#                   start_time=0,
#                   end_time=30)

# ===== extract frames for classification purposes =====
extract_frames(clipped_video_data_path=GlobalConfig.CLIPPED_VIDEO_DIR_PATH,
               num_frames=15,
               times=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29])


# ===== predict cat with vgg16 model =====
vgg16_model = load_vgg16_model()
cat_video_boolean_dict = preprocess_and_predict_frames(model=vgg16_model,
                                                       frame_dir=GlobalConfig.FRAMES_BASE_PATH)
print(cat_video_boolean_dict)
print('Success at running')









