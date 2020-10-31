# Imports
import pandas as pd

# YouTube api
from pytube import YouTube



def get_video(my_file):
    video = YouTube(url=my_file)
    video.streams.filter(file_extension='mp4').all()
    video.streams.get_by_itag(18).download()
    return video


def cat_popularity_classification(my_file):
    video = get_video(my_file)
    return video
