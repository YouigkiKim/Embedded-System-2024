# -*- coding: utf-8 -*-

import os

def rename_files_in_directory(directory,img_number, file_extension=".jpg" ):

    files = [f for f in os.listdir(directory) if f.endswith(file_extension)]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)))

    for index, filename in enumerate(files):
        new_filename = f"frame_{img_number+index:09d}{file_extension}"
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        os.rename(old_path, new_path)  
        print(f"Renamed '{filename}' to '{new_filename}'")


directory_path = 'dataset/05202302_straight'

rename_files_in_directory(directory_path,111110,file_extension=".jpg")
rename_files_in_directory(directory_path,189,file_extension=".jpg")
