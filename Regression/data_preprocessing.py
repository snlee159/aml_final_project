import sys
import pickle
import numpy as np
import pandas as pd


def extract_regression_features(use_sound_features=True, sound_feature_list=['bpm', 'beat_ratio'],

                                use_HOG=False, K=20,

                                use_pixel_change=False,
                                actual_pixel_change_feature=False,
                                categorical_pixel_change_feature=False,
                                single_scene_detection=False,

                                use_title=False,

                                labels_categorical=False, train_test_split_ratio=0.8):


    # load keys for relevant videos
    with open('../pkl/cat_video_boolean_dict.pkl', 'rb') as file:
        cat_video_boolean_dict = pickle.load(file)
    relevant_video_keys = []
    for key, value in cat_video_boolean_dict.items():
        if value == 1:
            relevant_video_keys.append(key)


    # list of dataframes to be merged
    feature_pd_list = []


    # load relevant sound features into pandas dataframe for relevant videos
    if use_sound_features:
        sound_features_raw_pd = pd.read_csv('../csv/sound_features - sound_features.csv')
        relevant_sound_features_pd = sound_features_raw_pd[[sound_feature for sound_feature in sound_feature_list]]
        feature_pd_list.append(relevant_sound_features_pd)

    # load HOG features for relevant videos
    if use_HOG:
        with open('../pkl/hog_feature_dict_K_{}.pkl'.format(K), 'rb') as file:
            hog_feature_dict = pickle.load(file)
        columns_list = []
        for i in range(1, K+1):
            columns_list.append('HOG_Feature_{}'.format(i))
        HOG_feature_pd = pd.DataFrame(columns=columns_list)
        for key in relevant_video_keys:
            hog_features_list = hog_feature_dict[key]
            HOG_feature_pd = HOG_feature_pd.append(pd.DataFrame([hog_features_list],
                                                                columns=columns_list),
                                                   ignore_index=True)
        feature_pd_list.append(HOG_feature_pd)


    # load pixel change feature
    if use_pixel_change:
        if actual_pixel_change_feature:
            with open('../pkl/pixel_change_feature_dict.pkl', 'rb') as file:
                pixel_change_feature_dict = pickle.load(file)
            columns_list = []
            for i in range(2, 16):
                columns_list.append('pixel_change_to_frame_{}'.format(i))
            pixel_change_features_pd = pd.DataFrame(columns=columns_list)
            for key in relevant_video_keys:
                actual_pixel_change_list = []
                pixel_change_frame_dict = pixel_change_feature_dict[key]
                for key, value in pixel_change_frame_dict.items():
                    actual_pixel_change_list.append(value)
                pixel_change_features_pd = pixel_change_features_pd.append(pd.DataFrame([actual_pixel_change_list],
                                                                                        columns=columns_list),
                                                                           ignore_index=True)
            feature_pd_list.append(pixel_change_features_pd)

        elif categorical_pixel_change_feature:
            with open('../pkl/pixel_change_dummy_feature_dict.pkl', 'rb') as file:
                pixel_change_dummy_feature_dict = pickle.load(file)
            columns_list = []
            for i in range(2, 16):
                columns_list.append('scene_change_to_frame_{}'.format(i))
            scene_change_features_pd = pd.DataFrame(columns=columns_list)
            for key in relevant_video_keys:
                categorical_pixel_change_list = []
                pixel_change_frame_dict = pixel_change_dummy_feature_dict[key]
                for key, value in pixel_change_frame_dict.items():
                    categorical_pixel_change_list.append(value)
                scene_change_features_pd = scene_change_features_pd.append(pd.DataFrame([categorical_pixel_change_list],
                                                                                        columns=columns_list),
                                                                           ignore_index=True)
            feature_pd_list.append(scene_change_features_pd)

        elif single_scene_detection:
            with open('../pkl/pixel_change_dummy_feature_dict.pkl', 'rb') as file:
                pixel_change_dummy_feature_dict = pickle.load(file)
            single_scene_change_feature_pd = pd.DataFrame(columns=['scene_change'])
            for key in relevant_video_keys:
                scene_change_list = [0]
                pixel_change_frame_dict = pixel_change_dummy_feature_dict[key]
                for key, value in pixel_change_frame_dict.items():
                    if value == 1:
                        scene_change_list[0] = 1
                        break
                single_scene_change_feature_pd = \
                    single_scene_change_feature_pd.append(pd.DataFrame([scene_change_list],
                                                                        columns=['scene_change']),
                                                           ignore_index=True)
            feature_pd_list.append(single_scene_change_feature_pd)


    # load title feature
    if use_title:
        with open('../pkl/video_title_feature_dict.pkl', 'rb') as file:
            title_feature_dict = pickle.load(file)
        title_features = ['street', 'owner', 'sound', 'keyboard', 'face', 'meme', 'dance', 'meow', 'mating', 'aww',
                          'adopt', 'baby', 'persian', 'massage', 'short', 'food', 'meet', 'lovely', 'memes', 'meowing']
        title_feature_pd = pd.DataFrame(columns=title_features)
        for key in relevant_video_keys:
            title_word_occ_list = []
            title = title_feature_dict[key]
            for word in title_features:
                if word in title:
                    title_word_occ_list.append(1)
                else:
                    title_word_occ_list.append(0)
            title_feature_pd = \
                title_feature_pd.append(pd.DataFrame([title_word_occ_list],
                                                     columns=title_features),
                                        ignore_index=True)
        feature_pd_list.append(title_feature_pd)


    # concatenate data frames to one big training data frame
    training_df = pd.concat([feature_pd for feature_pd in feature_pd_list], axis=1)
    training_data_numpy_array = training_df.to_numpy()


    # load view count as labels and create label array
    if labels_categorical:
        with open('../pkl/view_count_categorical_feature_dict.pkl', 'rb') as file:
            view_count_feature_dict = pickle.load(file)
    else:
        with open('../pkl/view_count_feature_dict.pkl', 'rb') as file:
            view_count_feature_dict = pickle.load(file)
    training_labels_array = []
    for key in relevant_video_keys:
        view_count = view_count_feature_dict[key]
        training_labels_array.append(view_count)
    training_labels_numpy_array = np.asarray(training_labels_array)


    # print shapes for control purposes
    print('Shape of training data: ', training_data_numpy_array.shape)
    print('Shape of training labels: ', training_labels_numpy_array.shape)


    # perform train test split
    split_index = int(train_test_split_ratio*training_data_numpy_array.shape[0])
    X_train = training_data_numpy_array[:split_index]
    y_train = training_labels_numpy_array[:split_index]
    X_test = training_data_numpy_array[split_index:]
    y_test = training_labels_numpy_array[split_index:]

    return X_train, y_train, X_test, y_test

