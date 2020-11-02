import requests
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta as td


def get_list_videos(keyword='cat'):
    # Initialize stuffs
    list_df = pd.DataFrame(columns=['videoId', 'title'])
    video_list = []
    title_list = []
    video_count = 0
    run_count = 0

    # We want 20,000 videos
    while video_count < 20000:
        # Find time range for getting around query limit
        publishedBefore = (dt.now() - td(days=(run_count) * 7)).isoformat("T") + "Z"
        publishedAfter = (dt.now() - td(days=(run_count + 1) * 7)).isoformat("T") + "Z"

        # Print to show progress
        print(f'Published Before: {publishedBefore}, video count: {video_count}, run count: {run_count}')

        # Set stop criteria and pagination
        stop = True
        nextPageToken = 'start'

        # If there are more pages, keep pulling
        while stop:
            if nextPageToken == 'start':
                response = requests.get(
                    f'https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=50&q={keyword}&type=video&videoCaption=none&videoDefinition=standard&videoDuration=short&publishedBefore={publishedBefore}&publishedAfter={publishedAfter}&key={google_token}')
                run_count += 1
            else:
                response = requests.get(
                    f'https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=50&q={keyword}&type=video&videoCaption=none&videoDefinition=standard&videoDuration=short&publishedBefore={publishedBefore}&publishedAfter={publishedAfter}&key={google_token}&pageToken={nextPageToken}')
                run_count += 1

            # Convert the response to json so it can be interpreted
            response = response.json()

            # Add data to the lists and iterate on the count
            for i, item in enumerate(response['items']):
                video_list += [item['id']['videoId']]
                title_list += [item['snippet']['title']]
                video_count += 1

            # Take action depending on the presence of a next page
            if len(response['items']) > 0:
                nextPageToken = response['nextPageToken']
            else:
                stop = False


    list_df['videoId'] = video_list
    list_df['title'] = title_list
    now = dt.strftime(dt.now(), "%Y%m%d_%H%M%S")
    list_df.to_csv(f'aml_final_project/csv/video_list_{now}.csv', index=False)


def get_views(saved_list_df):
    saved_list_df['view_count'] = 0
    saved_list_df['duration'] = ''
    for i, row in saved_list_df.iterrows():
        if i % 100 == 0:
            print(i)
        response = requests.get(
            f'https://www.googleapis.com/youtube/v3/videos?part=statistics,contentDetails&id={row["videoId"]}&key={google_token}')
        response = response.json()
        if len(response['items']) > 0:
            saved_list_df.loc[i, 'view_count'] = response['items'][0]['statistics']['viewCount']
            saved_list_df.loc[i, 'duration'] = response['items'][0]['contentDetails']['duration']

    now = dt.strftime(dt.now(), "%Y%m%d_%H%M%S")
    saved_list_df.to_csv(f'aml_final_project/csv/full_list_df_{now}.csv', index=False)