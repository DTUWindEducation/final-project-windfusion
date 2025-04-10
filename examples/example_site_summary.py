import sys
import os

# Dynamically add src/ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from finalproject.site_summary import SiteSummary


site = SiteSummary(site_index=1)
summary = site.summarize()