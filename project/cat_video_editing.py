import os
import pickle
import shutil
from project.global_config import GlobalConfig
from project.local_config import LocalConfig
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def limit_video_length(raw_video_data_path, start_time, end_time):
    '''
    :param raw_video_data_path: <str> path to directory with raw video data
    :param start_time: <int> second to start clipping
    :param end_time:  <int> second to end clipping
    :return:
    '''

    file_names = os.listdir(raw_video_data_path)
    for idx, file_name in enumerate(file_names):
        raw_video_path = os.path.join(raw_video_data_path, file_name)

        # https://stackoverflow.com/a/37323543
        ffmpeg_extract_subclip(filename=raw_video_path,
                               t1=start_time,
                               t2=end_time,
                               targetname=os.path.join(GlobalConfig.CLIPPED_VIDEO_DIR_PATH,
                                                       'Clip_{}.mp4'.format(idx)))



def extract_frames(clipped_video_data_path, num_frames, times):
    '''
    :param clipped_video_data_path: <str> path to clipped video files
    :param num_frames: <int> number of frames to extract
    :param times: <list> list of integers that represent the second at which frame shall be extracted
    :param video_number: <int> the number of the video from which frames are extracted
    :return:
    '''

    # catch illegal argument combinations
    if len(times) != num_frames:
        print('<num_frames> and length of <times> must be the same!')

    else:

        file_names = os.listdir(clipped_video_data_path)
        for file_name in file_names:

            # create directory for frames
            os.makedirs(os.path.join(LocalConfig.FRAMES_BASE_PATH,
                                     file_name[:-4]),
                        exist_ok=True)
            # load video
            video_clip = VideoFileClip(os.path.join(clipped_video_data_path, file_name))
            # extract and store frames
            for idx, time in enumerate(times):
                frame_path = os.path.join(GlobalConfig.FRAMES_BASE_PATH,
                                          file_name[:-4],
                                          'frame_{}.png'.format(idx+1))
                video_clip.save_frame(frame_path, str(time))
            # close loaded video and audio
            video_clip.reader.close()
            video_clip.audio.reader.close_proc()


def clean_dataset_with_dict(cat_video_boolean_dict):

    '''
    :param cat_video_boolean_dict: <dict> object that containes information which videos are cat videos
    :return:
    '''

    with open(cat_video_boolean_dict, 'rb') as file:
        cat_video_boolean_dict = pickle.load(file)

    source_basepath_video = GlobalConfig.CLIPPED_VIDEO_DIR_PATH
    destination_basepath_video = GlobalConfig.CLIPPED_VIDEO_DIR_CLEANED_PATH
    source_basepath_frames = GlobalConfig.FRAMES_BASE_PATH
    destination_basepath_frames = GlobalConfig.FRAMES_CLEANED_BASE_PATH


    for key, value in cat_video_boolean_dict.items():

        if value == 0:
            continue
        elif value == 1:
            # copy clipped videos
            source = os.path.join(source_basepath_video, str(key)+'.mp4')
            destination = os.path.join(destination_basepath_video, str(key)+'.mp4')
            shutil.copyfile(source, destination)

            # copy frames
            source = os.path.join(source_basepath_frames, key)
            os.makedirs(os.path.join(destination_basepath_frames, key))
            destination = os.path.join(destination_basepath_frames, key)
            frames = os.listdir(source)
            for frame in frames:
                frame_source = os.path.join(source, frame)
                frame_destination = os.path.join(destination, frame)
                shutil.copy(frame_source, frame_destination)
        else:
            print('Unknown value for key', key)




