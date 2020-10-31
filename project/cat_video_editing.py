import os
from project.local_config import LocalConfig
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def limit_video_length(video_path, start_time, end_time):
    '''
    :param video_path: <str> path to video
    :param start_time: <int> second to start clipping
    :param end_time:  <int> second to end clipping
    :return:
    '''

    # https://stackoverflow.com/a/37323543
    ffmpeg_extract_subclip(filename=video_path,
                           t1=start_time,
                           t2=end_time,
                           targetname=video_path[:-4]+'_limited_length'+video_path[-4:])



def extract_frames(video_path, video_number, num_frames, times):
    '''
    :param video_path: <str> path to video files
    :param num_frames: <int> number of frames to extract
    :param times: <list> list of integers that represent the second at which frame shall be extracted
    :param video_number: <int> the number of the video from which frames are extracted
    :return:
    '''

    # catch illegal argument combinations
    if len(times) != num_frames:
        print('<num_frames> and length of <times> must be the same!')

    else:
        # create directory for frames
        os.makedirs(os.path.join(LocalConfig.FRAMES_BASE_PATH,
                                 'clip_{}'.format(video_number)),
                    exist_ok=True)
        # load video
        video_clip = VideoFileClip(video_path)
        # extract and store frames
        for idx, time in enumerate(times):
            frame_path = os.path.join(LocalConfig.FRAMES_BASE_PATH,
                                      'clip_{}'.format(video_number),
                                      'frame_{}.png'.format(idx+1))
            video_clip.save_frame(frame_path, str(time))
        # close loaded video and audio
        video_clip.reader.close()
        video_clip.audio.reader.close_proc()



