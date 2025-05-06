"""
UNResolutionProcessor: PDFchunker.py

    splits processed text (with italic markings) into a list of quasi-sentences

"""
import re

def split_by_italics(text):
    # find all italicized phrases
    italic_phrases = re.findall(r'<i>(.*?)</i>', text)
    
    # split the text at these phrases
    chunks = re.split(r'<i>.*?</i>', text)
    
    # list to hold chunks
    final_chunks = []

    for i in range(len(italic_phrases)):
        # combine the phrase with the rest of the chunk
        combined_text = italic_phrases[i] + chunks[i+1]
        
        # find the last comma or semicolon before the italic phrase
        last_punctuation_index = max(combined_text.rfind(','), combined_text.rfind(';'))
        
        if last_punctuation_index != -1:
            # end of the res? no problem
            segment = combined_text[:last_punctuation_index + 1]
        else:
            # add as per usual
            segment = combined_text
        
        final_chunks.append(segment.strip())

    return final_chunks
