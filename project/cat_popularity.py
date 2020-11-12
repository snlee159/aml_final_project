# Imports
import pandas as pd

# YouTube api
from pytube import YouTube
#import cv2


def get_video(my_file):
    video = YouTube(url=my_file)
    video.streams.filter(file_extension='mp4').all()
    video.streams.get_by_itag(18).download()
    return video


def cat_popularity_classification(my_file):
    video = get_video(my_file)
    return video


def save_video_with_viewcount(videoId, view_count, video_folder):
    url = "https://www.youtube.com/watch?v=" + videoId
    video = YouTube(url)
    video.streams.get_by_itag(18).download(video_folder)
    file_name = re.sub(r"[.\"',\\/:|?<>#;~*]", "", video.title)
    old_file_name = file_name + ".mp4"
    new_file_name = file_name + " | Viewcount: " + view_count + ".mp4"
    os.rename(video_folder + old_file_name,video_folder + new_file_name)
    print("Successfully Saved: " + str(video.title))
    return video