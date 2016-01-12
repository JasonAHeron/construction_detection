from os import (
    listdir,
    path,
)

def get_all_from_dir(directory, file_format):
    return [path.abspath(file) for file in listdir(directory) if file.endswith(file_format)]
