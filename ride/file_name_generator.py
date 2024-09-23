from datetime import datetime
import os


def get_current_dir(folder: str = '') -> str:
    """
        Returns the path to the current directory, creating it if it does not exist.

        Parameters
        ------------
        folder : str, optional
            Optional subfolder to create within the current directory (default is an empty string)

        Returns
        --------
        str
            The path to the current directory
    """
    folder_name = 'clusters_results'
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    current_folder = datetime.today().strftime('%Y_%m_%d')
    path = os.path.join(folder_name, current_folder)
    if not os.path.isdir(path):
        os.mkdir(path)
    if folder!= '':
        path = os.path.join(path, folder)
        if not os.path.isdir(path):
            os.mkdir(path)
    return path


def generate_new_name(file_name: str, folder: str = '') -> str:
    """
        Generates a new file name by appending a number to the original name if the file already exists.

        Parameters
        ------------
        file_name : str
            The original file name
        folder : str, optional
            Optional subfolder to use (default is an empty string)

        Returns
        --------
        str
            The new file name
    """
    path = get_current_dir(folder)
    p = os.path.join(path, file_name)
    num = 1
    while os.path.isfile(p):
        p = os.path.join(path, file_name.split('.')[0] + '_' + str(num) + '.' + file_name.split('.')[1])
        num += 1
    return p