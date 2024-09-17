from datetime import datetime
import os


def get_current_dir(folder: str = '') -> str:
    folder_name = 'clusters_results'
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    current_folder = datetime.today().strftime('%Y_%m_%d')
    path = os.path.join(folder_name, current_folder)
    if not os.path.isdir(path):
        os.mkdir(path)
    if folder != '':
        path = os.path.join(path, folder)
        if not os.path.isdir(path):
            os.mkdir(path)
    return path


def generate_new_name(file_name: str, folder: str = '') -> str:
    path = get_current_dir(folder)
    p = os.path.join(path, file_name)
    num = 1
    while os.path.isfile(p):
        p = os.path.join(path, file_name.split('.')[0] + '_' + str(num) + '.' + file_name.split('.')[1])
        num += 1
    return p
