# Imports
import pandas as pd
from aml_final_project.pyAudioAnalysis1.pyAudioAnalysis import ShortTermFeatures
from aml_final_project.pyAudioAnalysis1.pyAudioAnalysis import audioBasicIO
import matplotlib.pyplot as plt
import os


def extract_wav(filename):
    os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}.wav'.format(filename, actual_filename))