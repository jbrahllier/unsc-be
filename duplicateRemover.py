"""
UNResolutionProcessor: duplicateRemover.py

    removes duplicates and adjusts clause IDs to be concurrent, keeps all other columns intact

"""
import pandas as pd

def remove_duplicates_and_adjust_ids(df, column_name, id_column):
    # remove duplicates
    df = df.drop_duplicates(subset=[column_name]).reset_index(drop=True)
    
    # adjust IDs
    df[id_column] = range(1, len(df) + 1)
    
    return df
