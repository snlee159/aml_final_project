import os
from project.global_config import GlobalConfig
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def load_vgg16_model():
    vgg16_model = VGG16()
    return vgg16_model


def preprocess_and_predict_frames(frame_dir, model):

    image_names = os.listdir(frame_dir)
    cat_frames_counter = 0
    for idx, image_name in enumerate(image_names):
        # preprocess image for vgg16
        image = load_img(os.path.join(frame_dir, image_name), target_size=(224, 224, 3))
        image_array = img_to_array(image)
        image_array_reshaped = image_array.reshape((1,
                                                    image_array.shape[0],
                                                    image_array.shape[1],
                                                    image_array.shape[2]))
        preprocessed_image = preprocess_input(image_array_reshaped)

        # let vgg16 model make a prediction
        prediction = model.predict(preprocessed_image)
        prediction_decoded = decode_predictions(prediction)
        print('Prediction frame {}:'.format(idx+1), prediction_decoded[0][0][1])

        if prediction_decoded[0][0][1] in GlobalConfig.VGG16_LABELS:
            cat_frames_counter += 1

    # decide whether its a cat video or not
    cat_non_cat_ratio = cat_frames_counter / len(image_names)
    print('Detected cats on {} % of the frames.'.format(cat_non_cat_ratio*100))
    if cat_non_cat_ratio >= 0.5:
        print('-->Cat video.')
    else:
        print('-->Probably not a cat video.')


