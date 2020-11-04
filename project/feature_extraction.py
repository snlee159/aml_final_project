import os
import pickle
import numpy as np
from project.global_config import GlobalConfig
from project.vgg16_cat_detector import load_vgg16_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array



def load_feature_extraction_model():
    feature_extraction_model = load_vgg16_model()
    return feature_extraction_model



def extract_pixel_change_feature(frame_dir):

    # create dictionary with respective features
    feature_dict = dict()

    # list Clip_0, Clip_1, Clip_2, ... directories
    subdir_names = os.listdir(frame_dir)

    for subdir in subdir_names:

        # update feature dict with clip level
        feature_dict.update({subdir: {}})

        # list frame_1, frame_2, frame_3, ... in each subdir
        frame_names = []
        for idx in range(0, len(os.listdir(os.path.join(frame_dir, subdir)))):
            frame_names.append('frame_{}.png'.format(idx + 1))

        for frame_idx in range(2, len(frame_names)+1):
            frame_i = load_img(os.path.join(frame_dir, subdir, 'frame_{}.png'.format(frame_idx)))
            frame_i_minus_1 = load_img(os.path.join(frame_dir, subdir, 'frame_{}.png'.format(frame_idx-1)))
            frame_i_array = img_to_array(frame_i)
            frame_i_minus_1_array = img_to_array(frame_i_minus_1)
            pixel_change = frame_i_array - frame_i_minus_1_array
            average_pixel_change = np.mean(np.abs(pixel_change))

            feature_dict.get(subdir).update({'change_to_frame_{}'.format(frame_idx): average_pixel_change})

    with open('pixel_change_feature.pkl', 'wb') as file:
        pickle.dump(feature_dict, file)




