# Imports
import pandas as pd

# YouTube api
from pytube import YouTube
import cv2

def get_video(file, nframes=10):
    video = YouTube(file)
    video.streams.filter(file_extension='mp4').all()
    return video.streams.get_by_itag(nframes).download()

def cat_popularity_classification():
    test = get_video('https://www.youtube.com/watch?v=C9O28ne6bG8')
    print(test)

def main():
    cat_popularity_classification()

if __name__ == "__main__":
    main()