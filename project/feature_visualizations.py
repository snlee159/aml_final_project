import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


def plot_pixel_change_feature_histogram(data_dict, n_bins=20):

    # load data dict
    with open(data_dict, 'rb') as file:
        pixel_change_feature_dict = pickle.load(file)

    # extract all pixel change values
    pixel_change_list = []
    for clip_key, pixel_dict in pixel_change_feature_dict.items():
        for frame_key, pixel_change_value in pixel_dict.items():
            pixel_change_list.append(pixel_change_value)

    # plot a histogram to see distribution of pixel change feature
    n, bins, patches = plt.hist(pixel_change_list, n_bins, facecolor='blue', alpha=0.5)

    # add average line to distribution
    plt.axvline(np.mean(np.asarray(pixel_change_list)), color='k', linestyle='dashed', linewidth=1)

    # configurate histogram
    plt.xlabel('Mean absolute pixel change -  bins')
    plt.ylabel('#Items in bin')
    plt.title(r'Histogram of Mean Absolute Pixel Change from Frame to Frame')
    plt.savefig('../visualizations/Pixel_change_histogram.png', dpi=200)
    plt.show()


def plot_view_count_feature_histogram(data_dict, n_bins=20, count_limit=10000000):

    # load data dict
    with open(data_dict, 'rb') as file:
        view_count_feature_dict = pickle.load(file)

    # extract all pixel change values
    view_count_list = []
    for clip_key, value in view_count_feature_dict.items():
        if value < count_limit:
            view_count_list.append(value)

    # plot a histogram to see distribution of pixel change feature
    n, bins, patches = plt.hist(view_count_list, n_bins, facecolor='blue', alpha=0.5, rwidth=0.9)

    # add average line to distribution
    plt.axvline(np.mean(np.asarray(view_count_list)), color='k', linestyle='dashed', linewidth=1)

    # configurate histogram
    plt.xlabel('View count bin')
    plt.ylabel('#Items in bin')
    plt.title(r'Histogram of View Count Feature')
    plt.savefig('../visualizations/View_count_histogram_{}_count_limit.png'.format(count_limit),
                dpi=200)
    plt.show()


