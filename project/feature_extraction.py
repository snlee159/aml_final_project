import os
import re
import pickle
import numpy as np
from skimage.feature import hog
from skimage.transform import resize
from skimage.io import imread
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from project.global_config import GlobalConfig
from project.vgg16_cat_detector import load_vgg16_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array



def load_feature_extraction_model():

    '''
    :return: loaded vgg16 model
    '''

    feature_extraction_model = load_vgg16_model()
    return feature_extraction_model



def extract_pixel_change_feature(frame_dir, make_dummy_feature=False, threshold=75):

    '''
    :param frame_dir: path to the dir in which the sub dirs with frames are stored
    :return:
    '''

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

            if make_dummy_feature:
                if average_pixel_change > threshold:
                    feature_dict.get(subdir).update({'change_to_frame_{}'.format(frame_idx): 1})
                else:
                    feature_dict.get(subdir).update({'change_to_frame_{}'.format(frame_idx): 0})

            else:
                feature_dict.get(subdir).update({'change_to_frame_{}'.format(frame_idx): average_pixel_change})

    if make_dummy_feature:
        with open('../pkl/pixel_change_dummy_feature_dict.pkl', 'wb') as file:
            pickle.dump(feature_dict, file)
    else:
        with open('pixel_change_feature_dict.pkl', 'wb') as file:
            pickle.dump(feature_dict, file)



def extract_viewcount_feature_from_raw_videos(cat_video_boolean_dict):

    '''
    :param cat_video_boolean_dict: dict that holds information what videos are actually cat videos
    :return:
    '''

    # load dict with relevant videos
    with open(cat_video_boolean_dict, 'rb') as file:
        cat_video_boolean_dict = pickle.load(file)

    # load indices of relevant cat videos
    relevant_cat_video_indices = []
    for key, value in cat_video_boolean_dict.items():
        if value == 1:
            key_regex = re.findall(r'Clip_\d*', str(key))[0]
            relevant_cat_video_indices.append(int(key_regex[5:]))

    # create new feature dictionary for view count
    view_count_feature_dict = dict()

    # extract view count of relevant cat videos from title of raw videos
    raw_video_titles = os.listdir(GlobalConfig.RAW_VIDEO_DIR_PATH)
    for video_idx, raw_video_title in enumerate(raw_video_titles):
        try:
            view_count_str = re.findall(r'Viewcount_ \d*', raw_video_title)[0]
            view_count = int(view_count_str[10:])
        except IndexError:
            view_count = 0

        if video_idx in relevant_cat_video_indices:
            view_count_feature_dict.update({'Clip_'+str(video_idx): view_count})

    # save view_count_feature dict
    with open('view_count_feature_dict.pkl', 'wb') as file:
        pickle.dump(view_count_feature_dict, file)

def make_view_count_feature_categorical(view_count_feature_dict):

    # load normal view_count_feature_dict
    with open(view_count_feature_dict, 'rb') as file:
        view_count_feature_dict = pickle.load(file)

    # set up new dict
    view_count_categorical_feature_dict = dict()
    for clip_key, view_count in view_count_feature_dict.items():
        if view_count < 1000:
            view_count_categorical_feature_dict.update({clip_key: 1})
        elif view_count < 5000:
            view_count_categorical_feature_dict.update({clip_key: 2})
        elif view_count < 20000:
            view_count_categorical_feature_dict.update({clip_key: 3})
        elif view_count < 100000:
            view_count_categorical_feature_dict.update({clip_key: 4})
        else:
            view_count_categorical_feature_dict.update({clip_key: 5})

    with open('../pkl/view_count_categorical_feature_dict.pkl', 'wb') as file:
        pickle.dump(view_count_categorical_feature_dict, file)



def extract_video_title_from_raw_videos(cat_video_boolean_dict):

    '''
    :param cat_video_boolean_dict: dict that holds information what videos are actually cat videos
    :return:
    '''

    # load dict with relevant videos
    with open(cat_video_boolean_dict, 'rb') as file:
        cat_video_boolean_dict = pickle.load(file)

    # load indices of relevant cat videos
    relevant_cat_video_indices = []
    for key, value in cat_video_boolean_dict.items():
        if value == 1:
            key_regex = re.findall(r'Clip_\d*', str(key))[0]
            relevant_cat_video_indices.append(int(key_regex[5:]))

    # create new feature dictionary for video title
    video_title_feature_dict = dict()

    # extract video title of relevant cat videos from title of raw videos
    raw_video_titles = os.listdir(GlobalConfig.RAW_VIDEO_DIR_PATH)
    for video_idx, raw_video_title in enumerate(raw_video_titles):
        try:
            video_title_str = re.findall(r'.*Viewcount_', raw_video_title)[0]
            video_title = video_title_str[:-12]
        except IndexError:
            video_title = 'No title given'

        # if video relevant, clean title and add to dict
        if video_idx in relevant_cat_video_indices:
            video_title_cleaned = ''
            for word in video_title.split():
                word_cleaned = re.sub('[^a-zA-Z0-9]+', '', word)
                video_title_cleaned = video_title_cleaned + word_cleaned + ' '
            video_title_cleaned = video_title_cleaned.lower().rstrip()
            video_title_feature_dict.update({'Clip_'+str(video_idx): video_title_cleaned})

    # save video_title_feature_dict
    with open('video_title_feature_dict.pkl', 'wb') as file:
        pickle.dump(video_title_feature_dict, file)




def hog_feature_extraction(num_clusters):

    # load images into one array so that they can be classified through K-means
    k_means_training_data_set = []
    for idx, clip_dir in enumerate(os.listdir(GlobalConfig.FRAMES_CLEANED_BASE_PATH)):
        print('Hog feature extraction for clip', idx)
        for frame in os.listdir(os.path.join(GlobalConfig.FRAMES_CLEANED_BASE_PATH, clip_dir)):
            try:
                # load and preprocess image
                image = imread(os.path.join(GlobalConfig.FRAMES_CLEANED_BASE_PATH, clip_dir, frame))
                image_resized = resize(image, (160, 240))

                # perform hog feature extraction
                hog_descriptor, _ = hog(image_resized, orientations=9, pixels_per_cell=(8, 8),
                                        cells_per_block=(2, 2), visualize=True, multichannel=True)
                k_means_training_data_set.append(hog_descriptor)
            except Exception as e:
                print('Failure for this frame..')


    # train k-means classifier on training data set
    kmeans_training_data = np.array(k_means_training_data_set)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(kmeans_training_data)

    # extract features for each image
    hog_feature_dict = dict()
    for clip_dir in os.listdir(GlobalConfig.FRAMES_CLEANED_BASE_PATH):
        hog_feature_dict.update({clip_dir: np.zeros(num_clusters)})
        for frame in os.listdir(os.path.join(GlobalConfig.FRAMES_CLEANED_BASE_PATH, clip_dir)):
            try:
                image = imread(os.path.join(GlobalConfig.FRAMES_CLEANED_BASE_PATH, clip_dir, frame))
                image_resized = resize(image, (160, 240))

                # perform hog feature extraction
                hog_descriptor, _ = hog(image_resized, orientations=9, pixels_per_cell=(8, 8),
                                        cells_per_block=(2, 2), visualize=True, multichannel=True)

                # make prediction with kmeans
                class_idx = kmeans.predict([hog_descriptor])

                # update respective feature vector
                hog_feature_dict.get(clip_dir)[class_idx] += 1
            except Exception as e:
                print('Failure for this frame..')


    # store hog_feature_dict
    with open('../pkl/hog_feature_dict_K_{}.pkl'.format(num_clusters), 'wb') as file:
        pickle.dump(hog_feature_dict, file)





