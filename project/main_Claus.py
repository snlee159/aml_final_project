from project.cat_popularity import cat_popularity_classification
from project.cat_video_editing import limit_video_length, extract_frames
from project.vgg16_cat_detector import load_vgg16_model
from project.vgg16_cat_detector import preprocess_and_predict_frames


# ===== Download video =====
# test = cat_popularity_classification('https://www.youtube.com/watch?v=XyNlqQId-nk')


# ===== edit video =====
# video_path = 'C:/Users/Feuring/PycharmProjects/aml_final_project/project/The funniest and most humorous cat videos ever! - Funny cat compilation.mp4'
# limit_video_length(video_path=video_path, start_time=0, end_time=30)

#video_path_limited_length='C:/Users/Feuring/PycharmProjects/aml_final_project/project/The funniest and most humorous cat videos ever! - Funny cat compilation_limited_length.mp4'
#extract_frames(video_path=video_path_limited_length,
#               video_number=1,
#               num_frames=5,
#               times=[1, 7, 13, 19, 25])


# ===== predict cat with vgg16 model =====
path = 'C:/Users/Feuring/PycharmProjects/aml_final_project/project/Frames/clip_2'

vgg16_model = load_vgg16_model()
preprocess_and_predict_frames(model=vgg16_model, frame_dir=path)

print('Sucess at running')









