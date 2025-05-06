"""
UNResolutionProcessor: ResolutionSlimmer.py

    basic text cleaning functions:
        - whitespace cleanup
        - delete newlines
        - replace non-ascii
        - application of the above in a filepath and string, where needed

"""
import re

def replace_white(text):
    return re.sub(r'\s+', ' ', text)

def replace_new_line(text):
    return text.replace('\n', '')

def replace_non_ascii(input_string, replacement=''):
    non_ascii_pattern = re.compile(r'[^\x00-\x7F]')
    result_string = non_ascii_pattern.sub(replacement, input_string)
    return result_string

def slimDown(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        content = ' '.join(line.strip() for line in lines)
        content = replace_white(content)
        content = replace_non_ascii(content)
    return content

def PDFslimDown(text):
    text = replace_white(text)
    text = replace_non_ascii(text)
    return text
    