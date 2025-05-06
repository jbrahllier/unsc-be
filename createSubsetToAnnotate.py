"""
UNResolutionProcessor: createSubsetToAnnotate.py

    creates a randomized (seed 42) sample of a dataframe (percentage variable) for annotation/verification

"""
import pandas as pd
import os
import ToExcel

def createSubsetToAnnotate(input_file, percentage):
    # load dataframe
    df = pd.read_csv(input_file)
    
    # shuffle based on 'clauseID'
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # how many rows to keep?
    retain_count = int(len(df) * (percentage / 100.0))
    
    # take subset
    df_subset = df.iloc[:retain_count]
    
    # create filename based on input filename
    base_name = os.path.basename(input_file)
    dir_name = os.path.dirname(input_file)
    new_file_name = f"annotated_{base_name}"
    new_file_path = os.path.join(dir_name, new_file_name)

    # save to .xlsx
    excel_file_name = f'{new_file_name}.xlsx' 
    excel_file_path = os.path.join(dir_name, excel_file_name)
    ToExcel.add_dataframe_to_excel(excel_file_path, 'Pre-Annotated Data', df_subset)
    print(f"Created new Excel file: {excel_file_path}")
    
    # save to .csv
    df_subset.to_csv(new_file_path, index=False)
    
    print(f"Pre-annotated subset saved to {new_file_path} as an .xlsx and .csv")
