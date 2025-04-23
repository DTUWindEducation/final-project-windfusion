# src/finalproject/path_utils.py

import os

def get_input_file_path(site_index):
    """
    Returns the absolute path to the input CSV for the given site index.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.abspath(os.path.join(here, '..', '..'))
    return os.path.join(project_root, 'inputs', f'Location{site_index}.csv')