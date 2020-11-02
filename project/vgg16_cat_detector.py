import os
from project.global_config import GlobalConfig
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def load_vgg16_model():
    vgg16_model = VGG16()
    return vgg16_model


def preprocess_and_predict_frames(model, frame_dir):

    # list Clip_0, Clip_1, Clip_2, ... directories
    subdir_names = os.listdir(frame_dir)

    # Keep track which video show cats and which ones don't
    cat_video_boolean_dict = dict()

    for subdir in subdir_names:

        print('=====================',subdir ,'=====================')

        # list frame_1, frame_2, frame_3, ... in each subdir
        frame_names = os.listdir(os.path.join(frame_dir, subdir))
        cat_frames_counter = 0
        for idx, frame_name in enumerate(frame_names):

            print('=========', frame_name, '=========')

            # preprocess each frame (image) for vgg16
            image = load_img(os.path.join(frame_dir, subdir, frame_name),
                             target_size=(360, 640, 3))
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
                prediction_decoded = decode_predictions(prediction)
                print('Prediction frame {} - (sub image {}):'.format(idx+1, sub_image_idx+1),
                      prediction_decoded[0][0][1])

                if prediction_decoded[0][0][1] in GlobalConfig.VGG16_LABELS:
                    print('Cat detected!')
                    cat_detected = True
                    cat_frames_counter += 1
                    break

        # decide whether its a cat video or not
        cat_non_cat_ratio = cat_frames_counter / len(frame_names)
        print('Detected cats on {} % of the frames.'.format(cat_non_cat_ratio*100))
        if cat_non_cat_ratio > 0.01:
            cat_video_boolean_dict.update({subdir: 1})
            print('-->Cat video.')
        else:
            cat_video_boolean_dict.update({subdir: 0})
            print('-->Probably not a cat video.')

    return cat_video_boolean_dict
