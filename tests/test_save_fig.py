import os
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from finalproject import save_figure

def test_save_figure():
    # Create a simple figure
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    filename = 'test_figure.png'
    subfolder = 'test_outputs'

    # Save the figure
    save_figure(fig, filename, subfolder=subfolder)

    # Build expected path
    here = os.path.abspath(os.path.dirname(__file__))  # /tests
    project_root = os.path.abspath(os.path.join(here, '..'))  # /final-project-windfusion
    expected_path = os.path.join(project_root, subfolder, filename)

    # Assert file was created
    assert os.path.exists(expected_path), f"Figure file not found at: {expected_path}"

    # Clean up (optional, if you want to avoid test artifacts)
    os.remove(expected_path)
    os.rmdir(os.path.join(project_root, subfolder))
