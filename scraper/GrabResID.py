"""
UNResolutionProcessor: GrabResID.py

    Grabs the res ID (and year, if wanted) from a [pdf] filename (was renamed according to the link on the UN website)
    NOTE: could potentially need to be reworked if the UN makes changes to their website
    
"""
import os
import re

def grab_resID(file):
    resID = int(re.sub(r'\.pdf$', '', os.path.basename(file))[:4])
    return resID

def grab_year(file):
    year = int(re.search(r'\((\d{4})\)', os.path.basename(file)).group(1))
    return year