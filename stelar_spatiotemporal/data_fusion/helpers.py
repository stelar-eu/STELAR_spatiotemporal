import os


def list_files_in_folder(folder_path,ending):
    list_of_files = []
    for foldername, subfolders, filenames in os.walk(folder_path,topdown=False):
        for filename in filenames:
            if filename.lower().endswith(ending.lower()):
                list_of_files.append(os.path.join(foldername,filename))

    assert list_of_files, f'No {ending} files found in folder{folder_path}'
    return list_of_files
