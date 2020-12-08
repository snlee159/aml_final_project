import os
import pickle
import numpy as np
from project.global_config import GlobalConfig
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def load_vgg16_model():

    '''
    :return: loaded vgg16 model
    '''

    vgg16_model = VGG16()
    return vgg16_model


def preprocess_and_predict_frames(model, frame_dir):

    '''
    :param model: model which detects cats (should be vgg16 trained on imagenet data set)
    :param frame_dir: path to directory which contains all the sub directories with the clips
    :return: dictionary that holds information which videos are cat videos
    '''

    # load class labels
    with open('../pkl/vgg16_labels.pkl', 'rb') as file:
        vgg16_labels = pickle.load(file)

    # list Clip_0, Clip_1, Clip_2, ... directories
    subdir_names = os.listdir(frame_dir)
    # subdir_names = []
    # for idx in range(0, len(os.listdir(frame_dir))):
    #     subdir_names.append('Clip_{}'.format(idx))

    # Keep track which video show cats and which ones don't
    cat_video_boolean_dict = dict()

    for idx, subdir in enumerate(subdir_names):

        print('#', idx)
        print('=====================', subdir, '=====================')

        # list frame_1, frame_2, frame_3, ... in each subdir
        frame_names = os.listdir(os.path.join(frame_dir, subdir))
        # frame_names = []
        # for idx in range(0, len(os.listdir(os.path.join(frame_dir, subdir)))):
        #     frame_names.append('frame_{}.png'.format(idx+1))
        cat_frames_counter = 0
        cat_frame_list = []
        for idx, frame_name in enumerate(frame_names):

            print('=========', frame_name, '=========')

            # preprocess each frame (image) for vgg16
            if os.path.exists(os.path.join(frame_dir, subdir, frame_name)):
                image = load_img(os.path.join(frame_dir, subdir, frame_name),
                                 target_size=(360, 640, 3))
            else:
                print('Frames do not exist for this clip.')
                continue
            image_array = img_to_array(image)
            image_array_reshaped = image_array.reshape((1,
                                                        image_array.shape[0],
                                                        image_array.shape[1],
                                                        image_array.shape[2]))
            # divide image into sub images
            sub_image_11 = image_array_reshaped[:, :224, :224, :]
            sub_image_12 = image_array_reshaped[:, :224, 224:448, :]
            sub_image_13 = image_array_reshaped[:, :224, 416:, :]
            sub_image_21 = image_array_reshaped[:, 136:, :224, :]
            sub_image_22 = image_array_reshaped[:, 136:, 224:448, :]
            sub_image_23 = image_array_reshaped[:, 136:, 416:, :]
            sub_images = [sub_image_11, sub_image_12, sub_image_13, sub_image_21, sub_image_22, sub_image_23]
            cat_detected = False
            for sub_image_idx, sub_image in enumerate(sub_images):
                preprocessed_image = preprocess_input(sub_image)

                # let vgg16 model make a prediction for every sub image
                prediction = model.predict(preprocessed_image)
                predicted_class = np.argmax(np.asarray(prediction[0]))
                print('Prediction frame {} - (sub image {}):'.format(idx+1, sub_image_idx+1),
                      predicted_class, '-', vgg16_labels[predicted_class])

                if predicted_class in GlobalConfig.VGG16_CAT_LABEL_INDICES:
                    print('Cat detected!')
                    cat_frame_list.append(frame_name)
                    cat_frames_counter += 1
                    break


        # decide whether its a cat video or not
        try:
            cat_non_cat_ratio = cat_frames_counter / len(frame_names)
        except ZeroDivisionError:
            print('No cat frames exist for this clip.')
            cat_non_cat_ratio = 0
        print('Detected cats on {} % of the frames.'.format(cat_non_cat_ratio*100))
        if cat_non_cat_ratio > 0.2:
            cat_video_boolean_dict.update({subdir: cat_frame_list})
            print('-->Cat video.')
        else:
            # cat_video_boolean_dict.update({subdir: 0})
            print('-->Probably not a cat video.')

    with open('../pkl_final/cat_video_boolean_dict.pkl', 'wb') as file:
        pickle.dump(cat_video_boolean_dict, file)

    return cat_video_boolean_dict
