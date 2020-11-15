import pickle
import numpy as np
import pandas as pd


def extract_regression_features(sound_feature_list=['bpm', 'beat_ratio'],
                                pixel_change=True, pixel_change_dummy=False,
                                reduce_pixel_change_to_one_variable=False,
                                labels_categorical=False, train_test_split_ratio=0.8):

    # load relevant sound features into pandas dataframe
    sound_features_raw_pd = pd.read_csv('../csv/sound_features - sound_features.csv')
    relevant_sound_features_pd = sound_features_raw_pd[[sound_feature for sound_feature in sound_feature_list]]

    if pixel_change:
        if pixel_change_dummy:
            # load pixel change feature into pandas dataframe
            with open('../pkl/pixel_change_dummy_feature_dict.pkl', 'rb') as file:
                pixel_change_feature_dict = pickle.load(file)
        else:
            # load pixel change feature into pandas dataframe
            with open('../pkl/pixel_change_feature_dict.pkl', 'rb') as file:
                pixel_change_feature_dict = pickle.load(file)

        if reduce_pixel_change_to_one_variable:
            scene_change_feature_list = []
            for clip_key, frame_dict in pixel_change_feature_dict.items():
                scene_change_detected = False
                for frame_key, pixel_change in frame_dict.items():
                    if pixel_change == 1:
                        scene_change_detected = True
                        break
                if scene_change_detected:
                    scene_change_feature_list.append([1])
                else:
                    scene_change_feature_list.append([0])

            # create data frame with information
            pixel_change_features_pd = pd.DataFrame(columns=['Scene Change'], data=scene_change_feature_list)

        else:
            columns_list = []
            for i in range(2, 16):
                columns_list.append('change_to_frame_{}'.format(i))
            pixel_change_features_pd = pd.DataFrame(columns=columns_list)

            for clip_key, value_dict in pixel_change_feature_dict.items():
                pixel_feature_list = []
                for frame_key, pixel_change in value_dict.items():
                    pixel_feature_list.append(pixel_change)
                pixel_change_features_pd = \
                    pixel_change_features_pd.append(pd.DataFrame([pixel_feature_list],
                                                                 columns=columns_list),
                                                    ignore_index=True)
        # merge both dataframes
        training_df = pd.concat([relevant_sound_features_pd, pixel_change_features_pd], axis=1)
        training_data_numpy_array = training_df.to_numpy()


    else:
        training_data_numpy_array = relevant_sound_features_pd.to_numpy()


    # load view count as labels and create label array
    if labels_categorical:
        with open('../pkl/view_count_categorical_feature_dict.pkl', 'rb') as file:
            view_count_feature_dict = pickle.load(file)
    else:
        with open('../pkl/view_count_feature_dict.pkl', 'rb') as file:
            view_count_feature_dict = pickle.load(file)

    training_labels_array = []
    for key, value in view_count_feature_dict.items():
        training_labels_array.append(value)
    training_labels_numpy_array = np.asarray(training_labels_array)

    print('Shape of training data: ', training_data_numpy_array.shape)
    print('Shape of training labels: ', training_labels_numpy_array.shape)


    # perform train test split
    split_index = int(train_test_split_ratio*training_data_numpy_array.shape[0])
    X_train = training_data_numpy_array[:split_index]
    y_train = training_labels_numpy_array[:split_index]

    X_test = training_data_numpy_array[split_index:]
    y_test = training_labels_numpy_array[split_index:]

    return X_train, y_train, X_test, y_test

