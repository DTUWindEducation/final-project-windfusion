import os
from finalproject import get_input_file_path

def test_get_input_file_path():
    site_index = 1
    here = os.path.abspath(os.path.dirname(__file__))  # .../tests
    project_root = os.path.abspath(os.path.join(here, '..'))  # .../final-project-windfusion
    expected_path = os.path.join(project_root, 'inputs', f'Location{site_index}.csv')

    result = get_input_file_path(site_index)

    assert result == expected_path
    assert os.path.exists(result), f"Expected file does not exist: {result}"
