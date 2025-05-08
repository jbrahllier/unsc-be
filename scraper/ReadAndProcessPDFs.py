"""
UNResolutionProcessor: ReadAndProcessPDFs.py

    processes all the res pdf files in a directory and outputs the dataframe

"""

import pandas as pd
import re
import os
from itertools import count
import GrabResID
import ResolutionSlimmer
import PDFextractor
import PDFchunker
import ProcessHelpers

def is_valid_clause(clause):
    # no special characters allowed
    return bool(clause) and not re.match(r'^\d+$', clause) and not re.match(r'^\s*$', clause)

def collapse_and_trim_clause(clause):
    # collapse and trim string
    return ' '.join(clause).strip()

def read_and_process_paragraphs(file_path, failed_pdfs, exclude_annex, annex_pdfs):
    resID = GrabResID.grab_resID(file_path) # for res ID column
    day, month, year = PDFextractor.extract_date_from_pdf(file_path) # for date column(s)

    bboxes, annex_pdf = PDFextractor.get_main_content_bboxes(file_path, exclude_annex) # extract file bboxes
    if annex_pdf is not None:
        annex_pdfs.append(annex_pdf) # is annex pdf? cool

    try:
        # extract processed text from processed bboxes
        formatted_text = PDFextractor.extract_text_with_italics(file_path, bboxes) 
    except Exception as e:
        ProcessHelpers.safe_print(f"Error extracting text with italics for {file_path}: {e}")
        failed_pdfs.append(f"S/RES/{GrabResID.grab_resID(file_path)}") # if failed, add to the list and notify
        return pd.DataFrame({'clause': [], 'resID': [], 'year': [], 'month': [], 'day': []})

    reformatted_text = ResolutionSlimmer.PDFslimDown(formatted_text) # weight watchers
    clauses = PDFchunker.split_by_italics(reformatted_text) # extract list of quasi-sentences

    clauses = [clause for clause in clauses if is_valid_clause(clause)] # double-check validity

    clauses = [collapse_and_trim_clause([clause]) for clause in clauses] # redo slimdown (for each clause, not redundant)

    # dup the res specific info
    dup_resID = [resID] * len(clauses)
    dup_year = [year] * len(clauses)
    dup_month = [month] * len(clauses)
    dup_day = [day] * len(clauses)

    # add columns to the frame
    df = pd.DataFrame({'clause': clauses, 'resID': dup_resID, 'year': dup_year, 'month': dup_month, 'day': dup_day})

    return df

def process_all_files(folder_path, failed_pdfs, exclude_annex, annex_pdfs):
    files = [os.path.join(root, name)
    for root, _, files in os.walk(folder_path)
        for name in files
            if name.endswith(".pdf")]

    unique_clause_num = count(1)

    final_data = pd.DataFrame(columns=['clause', 'clauseID', 'resID', 'year', 'month', 'day'])

    # process all files in the directory
    for file in files:
        df = read_and_process_paragraphs(file, failed_pdfs, exclude_annex, annex_pdfs)
        df['clauseID'] = [next(unique_clause_num) for _ in range(len(df))] # assign 'clauseID' on a directory (not res) basis
        final_data = pd.concat([final_data, df], ignore_index=True)

    return final_data