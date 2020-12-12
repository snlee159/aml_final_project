import os
import shutil
import pickle
import numpy as np
import matplotlib.pyplot as plt

from random import shuffle
from sklearn.metrics import precision_recall_fscore_support
from project.global_config import GlobalConfig

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array





def create_binary_dataset(view_threshold=10000, view_count_dict='../pkl_final/view_count_feature_dict.pkl'):

    # create directories for training data set
    os.makedirs(os.path.join(GlobalConfig.DATA_BASE_PATH, 'CNN_dataset'), exist_ok=True)
    os.makedirs(os.path.join(GlobalConfig.DATA_BASE_PATH, 'CNN_dataset', 'popular'), exist_ok=True)
    os.makedirs(os.path.join(GlobalConfig.DATA_BASE_PATH, 'CNN_dataset', 'unpopular'), exist_ok=True)

    # define paths
    popular_class_path = os.path.join(GlobalConfig.DATA_BASE_PATH, 'CNN_dataset', 'popular')
    unpopular_class_path = os.path.join(GlobalConfig.DATA_BASE_PATH, 'CNN_dataset', 'unpopular')

    # load view_count_dict
    with open(view_count_dict, 'rb') as file:
        view_count_dict = pickle.load(file)

    # iterate over keys to determine to which directory frames should be copied
    popular_cat_frame_counter = 1
    unpopular_cat_frame_counter = 1
    for video_key, view_count_value in view_count_dict.items():

        # path to frames
        frames_path = os.path.join(GlobalConfig.FRAMES_CLEANED_BASE_PATH, video_key)

        # frame names
        frame_names = os.listdir(frames_path)

        for frame_name in frame_names:
            source = os.path.join(frames_path, frame_name)
            if view_count_value >= view_threshold:
                destination = os.path.join(popular_class_path, 'popular_cat_{}.png'.format(popular_cat_frame_counter))
                popular_cat_frame_counter += 1
            else:
                destination = os.path.join(unpopular_class_path, 'unpopular_cat_{}.png'.format(unpopular_cat_frame_counter))
                unpopular_cat_frame_counter += 1
            shutil.copyfile(source, destination)


def create_train_val_test_sets():

    # create directories
    os.makedirs(os.path.join(GlobalConfig.DATA_BASE_PATH, 'CNN_dataset', 'train'), exist_ok=True)
    os.makedirs(os.path.join(GlobalConfig.DATA_BASE_PATH, 'CNN_dataset', 'validation'), exist_ok=True)
    os.makedirs(os.path.join(GlobalConfig.DATA_BASE_PATH, 'CNN_dataset', 'test'), exist_ok=True)
    os.makedirs(os.path.join(GlobalConfig.DATA_BASE_PATH, 'CNN_dataset', 'train', 'popular'), exist_ok=True)
    os.makedirs(os.path.join(GlobalConfig.DATA_BASE_PATH, 'CNN_dataset', 'train', 'unpopular'), exist_ok=True)
    os.makedirs(os.path.join(GlobalConfig.DATA_BASE_PATH, 'CNN_dataset', 'validation', 'popular'), exist_ok=True)
    os.makedirs(os.path.join(GlobalConfig.DATA_BASE_PATH, 'CNN_dataset', 'validation', 'unpopular'), exist_ok=True)
    os.makedirs(os.path.join(GlobalConfig.DATA_BASE_PATH, 'CNN_dataset', 'test', 'popular'), exist_ok=True)
    os.makedirs(os.path.join(GlobalConfig.DATA_BASE_PATH, 'CNN_dataset', 'test', 'unpopular'), exist_ok=True)

    # define paths
    popular_class_path = os.path.join(GlobalConfig.DATA_BASE_PATH, 'CNN_dataset', 'popular')
    unpopular_class_path = os.path.join(GlobalConfig.DATA_BASE_PATH, 'CNN_dataset', 'unpopular')

    # load list of frames in each class
    popular_class_frames = os.listdir(popular_class_path)
    unpopular_class_frames = os.listdir(unpopular_class_path)


    # determine train, validation and test set
    train_val_split_idx_popular = int(len(popular_class_frames)*0.8)
    train_val_split_idx_unpopular = int(len(unpopular_class_frames)*0.8)
    val_test_split_idx_popular = int(len(popular_class_frames)*0.9)
    val_test_split_idx_unpopular = int(len(unpopular_class_frames)*0.9)

    # move popular cat frames
    for popular_cat_frame in popular_class_frames[:train_val_split_idx_popular]:
        source = os.path.join(popular_class_path, popular_cat_frame)
        destination = os.path.join(GlobalConfig.DATA_BASE_PATH, 'CNN_dataset', 'train', 'popular', popular_cat_frame)
        shutil.move(source, destination)
    for popular_cat_frame in popular_class_frames[train_val_split_idx_popular:val_test_split_idx_popular]:
        source = os.path.join(popular_class_path, popular_cat_frame)
        destination = os.path.join(GlobalConfig.DATA_BASE_PATH, 'CNN_dataset', 'validation', 'popular', popular_cat_frame)
        shutil.move(source, destination)
    for popular_cat_frame in popular_class_frames[val_test_split_idx_popular:]:
        source = os.path.join(popular_class_path, popular_cat_frame)
        destination = os.path.join(GlobalConfig.DATA_BASE_PATH, 'CNN_dataset', 'test', 'popular', popular_cat_frame)
        shutil.move(source, destination)

    # move unpopular cat frames
    for unpopular_cat_frame in unpopular_class_frames[:train_val_split_idx_unpopular]:
        source = os.path.join(unpopular_class_path, unpopular_cat_frame)
        destination = os.path.join(GlobalConfig.DATA_BASE_PATH, 'CNN_dataset', 'train', 'unpopular', unpopular_cat_frame)
        shutil.move(source, destination)
    for unpopular_cat_frame in unpopular_class_frames[train_val_split_idx_unpopular:val_test_split_idx_unpopular]:
        source = os.path.join(unpopular_class_path, unpopular_cat_frame)
        destination = os.path.join(GlobalConfig.DATA_BASE_PATH, 'CNN_dataset', 'validation', 'unpopular', unpopular_cat_frame)
        shutil.move(source, destination)
    for unpopular_cat_frame in unpopular_class_frames[val_test_split_idx_unpopular:]:
        source = os.path.join(unpopular_class_path, unpopular_cat_frame)
        destination = os.path.join(GlobalConfig.DATA_BASE_PATH, 'CNN_dataset', 'test', 'unpopular', unpopular_cat_frame)
        shutil.move(source, destination)




#=========================



def temp_method(view_threshold=10000, view_count_dict='../pkl_final/view_count_feature_dict.pkl'):


    # define paths
    popular_class_path = os.path.join(GlobalConfig.DATA_BASE_PATH, 'CNN_dataset', 'test', 'popular')
    unpopular_class_path = os.path.join(GlobalConfig.DATA_BASE_PATH, 'CNN_dataset', 'test', 'unpopular')


    # load view_count_dict
    with open(view_count_dict, 'rb') as file:
        view_count_dict = pickle.load(file)

    # iterate over keys to determine to which directory frames should be copied
    popular_cat_frame_counter = 1
    unpopular_cat_frame_counter = 1
    for video_key, view_count_value in view_count_dict.items():

        video_num = int(video_key[5:])

        if video_num < 16311:
            continue
        else:

            # path to frames
            frames_path = os.path.join(GlobalConfig.FRAMES_CLEANED_BASE_PATH, video_key)

            # frame names
            frame_names = os.listdir(frames_path)

            for frame_name in frame_names:
                source = os.path.join(frames_path, frame_name)
                if view_count_value >= view_threshold:
                    destination = os.path.join(popular_class_path, 'popular_cat_{}.png'.format(popular_cat_frame_counter))
                    popular_cat_frame_counter += 1
                else:
                    destination = os.path.join(unpopular_class_path, 'unpopular_cat_{}.png'.format(unpopular_cat_frame_counter))
                    unpopular_cat_frame_counter += 1
                shutil.copyfile(source, destination)


#===============================================



class CNN_model:

    def __init__(self, model_name):
        self.train_dir_path=os.path.join(GlobalConfig.DATA_BASE_PATH, 'CNN_dataset', 'train')
        self.val_dir_path=os.path.join(GlobalConfig.DATA_BASE_PATH, 'CNN_dataset', 'validation')
        self.test_dir_path=os.path.join(GlobalConfig.DATA_BASE_PATH, 'CNN_dataset', 'test')
        self.model_name = model_name
        self.model_path=os.path.join(GlobalConfig.PROGRAMMING_OUTPUTS_BASE_PATH, model_name)
        os.makedirs(self.model_path, exist_ok=True)


    def set_up_model(self, use_transfer_learning):

        if use_transfer_learning:

            conv_base = VGG16(weights='imagenet',
                              include_top=False,
                              input_shape=(224, 224, 3))
            conv_base.trainable = False
            # set_trainable = False
            # for layer in conv_base.layers:
            #     if 'block4' in layer.name or 'block5' in layer.name:
            #         set_trainable = True
            #     if set_trainable:
            #         layer.trainable = True
            #     else:
            #         layer.trainable = False
            self.model = models.Sequential()
            self.model.add(conv_base)
            self.model.add(layers.Flatten())
            self.model.add(layers.Dense(256, activation='relu'))
            self.model.add(layers.Dense(1, activation='sigmoid'))

        else:
            self.model = models.Sequential()
            self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
            self.model.add(layers.Conv2D(32, (3, 3), activation='relu'))
            self.model.add(layers.MaxPooling2D((2, 2)))
            self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            self.model.add(layers.MaxPooling2D((2, 2)))
            self.model.add(layers.Conv2D(128, (3, 3), activation='relu'))
            self.model.add(layers.Conv2D(128, (3, 3), activation='relu'))
            self.model.add(layers.MaxPooling2D((2, 2)))
            self.model.add(layers.Conv2D(128, (3, 3), activation='relu'))
            self.model.add(layers.MaxPooling2D((2, 2)))
            self.model.add(layers.Flatten())
            self.model.add(layers.Dense(512, activation='relu'))
            self.model.add(layers.Dense(2, activation='sigmoid'))

        print(self.model.summary())

        self.model.compile(loss='binary_crossentropy',
        optimizer=optimizers.RMSprop(lr=1e-4),
        metrics=['acc'])

        print('Model created.')



    def set_up_data_generators(self, use_data_augmentation):

        if use_data_augmentation:
            train_datagen = ImageDataGenerator(rescale=1./255,
                                               rotation_range=40,
                                               width_shift_range=0.2,
                                               height_shift_range=0.2,
                                               shear_range=0.2,
                                               zoom_range=0.2,
                                               horizontal_flip=True,
                                               fill_mode='nearest')
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)

        test_datagen = ImageDataGenerator(rescale=1./255)

        self.train_generator = train_datagen.flow_from_directory(self.train_dir_path,
                                                                    target_size=(224, 224),
                                                                    batch_size=23,
                                                                    class_mode='categorical')
        self.validation_generator = test_datagen.flow_from_directory(self.val_dir_path,
                                                                        target_size=(224, 224),
                                                                        batch_size=23,
                                                                        class_mode='categorical')
        print('Data generators created.')

    def train_model(self, save_model=True):
        self.history = self.model.fit_generator(self.train_generator,
                                                  steps_per_epoch=100,
                                                  epochs=50,
                                                  validation_data=self.validation_generator,
                                                  validation_steps=50,
                                                  use_multiprocessing=False,
                                                  workers=8)
        print('Training completed')

        if save_model==True:
            self.model.save(os.path.join(self.model_path, self.model_name+'.h5'))
            print('Model saved.')


    def plot_model_history(self):
        acc = self.history.history['acc']
        val_acc = self.history.history['val_acc']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.savefig(os.path.join(self.model_path, 'Training_and_validation_accuracy.png'), dpi=400)

        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.savefig(os.path.join(self.model_path, 'Training_and_validation_loss.png'), dpi=400)

        print('Model history plotted.')


def evaluate_model(model_name):

    model = load_model(os.path.join(GlobalConfig.PROGRAMMING_OUTPUTS_BASE_PATH, model_name, model_name+'.h5'))
    test_dir = os.path.join(GlobalConfig.DATA_BASE_PATH, 'CNN_dataset', 'test')

    test_popular_dir = os.path.join(test_dir, 'popular')
    test_unpopular_dir = os.path.join(test_dir, 'unpopular')


    # popular predictions
    actual_labels_popular = []
    predictions_popular = []
    for popular_image in os.listdir(test_popular_dir)[:20]:

        image =load_img(os.path.join(test_popular_dir, popular_image), target_size=(224, 224, 3))
        image_array = img_to_array(image)
        image_array = image_array / 255
        image_array_reshaped = image_array.reshape((1,
                                                    image_array.shape[0],
                                                    image_array.shape[1],
                                                    image_array.shape[2]))
        prediction = model.predict(image_array_reshaped)
        if prediction[0][0] >= 0.5:
            predictions_popular.append(1)
        else:
            predictions_popular.append(0)
        actual_labels_popular.append(1)

    # unpopular predictions
    actual_labels_unpopular = []
    predictions_unpopular = []
    for unpopular_image in os.listdir(test_unpopular_dir)[:20]:

        image =load_img(os.path.join(test_unpopular_dir, unpopular_image), target_size=(224, 224, 3))
        image_array = img_to_array(image)
        image_array = image_array / 255
        image_array_reshaped = image_array.reshape((1,
                                                    image_array.shape[0],
                                                    image_array.shape[1],
                                                    image_array.shape[2]))
        prediction = model.predict(image_array_reshaped)
        if prediction[0][1] > 0.5:
            predictions_unpopular.append(0)
        else:
            predictions_unpopular.append(1)
        actual_labels_unpopular.append(0)

    # combine arrays
    labels = np.concatenate((actual_labels_popular, actual_labels_unpopular), axis=0)
    predictions = np.concatenate((predictions_popular, predictions_unpopular), axis=0)

    precision, recall, _, _ = precision_recall_fscore_support(y_pred=predictions, y_true=labels, average='binary')
    f1 = (2*precision*recall) / (precision + recall)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1:', f1)

    # test_datagen = ImageDataGenerator(rescale=1./255)
    #
    # test_generator = test_datagen.flow_from_directory(test_dir,
    #                                                   target_size=(224, 224),
    #                                                   batch_size=23,
    #                                                   class_mode='categorical')
    # test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
    # print('test acc:', test_acc)










