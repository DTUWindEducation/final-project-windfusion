import os
import sys
import pytest

# Add src/ to the path so we can import the class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from finalproject.site_summary import SiteSummary

def test_create_site_summary():
    """
    Test that a SiteSummary object can be created for site 1
    and that data is loaded successfully.
    """
    site = SiteSummary(site_index=1)
    assert site.site_index == 1
    assert not site.data.empty  # DataFrame shouldn't be empty

def test_summarize_keys():
    """
    Test that the summarize method returns all expected keys.
    """
    site = SiteSummary(site_index=1)
    summary = site.summarize()

    expected_keys = {
        "Site Index",
        "Start Time",
        "End Time",
        "Total Hours",
        "Mean Wind Speed @100m (m/s)",
        "Mean Power Output (normalized)",
        "Max Power Output",
        "Min Power Output",
        "Mean Temperature (C)",
        "Mean Relative Humidity (%)"
    }

    assert set(summary.keys()) == expected_keys

def test_summarize_values():
    """
    Test that returned summary values are of correct types.
    """
    site = SiteSummary(site_index=1)
    summary = site.summarize()

    assert isinstance(summary["Total Hours"], int)
    assert isinstance(summary["Mean Wind Speed @100m (m/s)"], float)
    assert 0 <= summary["Mean Power Output (normalized)"] <= 1
