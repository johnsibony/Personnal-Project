"""
Get password from environment variable. 
"""

# Author: John Sibony <john.sibony@hotmail.fr>

import os

def get_link_engine():
    """Return the link of the Database which is stored in an environment variable."""
    link_engine = os.environ.get('amazon_database')
    link_engine = 'postgresql://quant:jxMHYv0pntMJQz2k5a8c@tcgdwh.csx3ywdr9ixy.us-east-1.rds.amazonaws.com:5432/dwh'
    return link_engine