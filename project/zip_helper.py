import os
import shutil
import zipfile


zip_base_path = 'C:/Users/cpfeu/Data/AML_Final_Project/zip_NEW'
zip_destination_temp = 'C:/Users/cpfeu/Data/AML_Final_Project/unzipped_temp_2'
zip_destination = 'C:/Users/cpfeu/Data/AML_Final_Project/raw_video_data'

all_zip_files = os.listdir(zip_base_path)

for zip_file in all_zip_files:
    with zipfile.ZipFile(os.path.join(zip_base_path, zip_file), 'r') as zip_ref:
        zip_ref.extractall(zip_destination_temp)


# for unzipped_dir in os.listdir(zip_destination_temp):
#     for file in os.listdir(os.path.join(zip_destination_temp, unzipped_dir)):
#
#         # Source path
#         source = os.path.join(zip_destination_temp, unzipped_dir, file)
#
#         # Destination path
#         destination = os.path.join(zip_destination, file)
#
#         # Copy the content of
#         # source to destination
#         dest = shutil.copyfile(source, destination)