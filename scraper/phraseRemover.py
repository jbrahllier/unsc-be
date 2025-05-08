"""
UNResolutionProcessor: phraseRemover.py

    removes phrases and adjusts clause IDs to be concurrent, keeps all other columns intact

"""
import pandas as pd

def remove_strings_and_adjust_ids(df, column_name, strings_to_remove, id_column):
    # remove rows containing the specified strings
    df = df[~df[column_name].isin(strings_to_remove)].reset_index(drop=True)
    
    # adjust IDs
    df[id_column] = range(1, len(df) + 1)
    
    return df
