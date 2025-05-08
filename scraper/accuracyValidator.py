"""
import pandas as pd
import os 
import ToExcel

def classify_clauses(file_path):
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Ensure that the necessary columns exist
    if 'clause' not in df.columns:
        raise ValueError("The Excel file must have a 'clause' column")
    
    # Function to check if clause begins with a capital letter and ends with a period, semicolon, or comma
    def check_clause(clause):
        if clause and isinstance(clause, str):
            return 1 if clause[0].isupper() and clause[-1] in ['.', ';', ','] else 0
        return 0
    
    # Apply the function to each clause and store the result in the 'correct' column
    df['correct'] = df['clause'].apply(check_clause)
    
    # Save the modified DataFrame back to an Excel file
    output_file = file_path.replace('.xlsx', '_classified.xlsx')
    df.to_excel(output_file, index=False)
    
    return output_file

# Example usage
file_path = '/Users/jacobcollier/Desktop/Summer_2024/BERT/UNResProcessorPDF/UNResolutionData/annotated_UNResolutionData_2024_07_29_0409PM.csv.xlsx'  # Replace with actual file path
folder_name = '/Users/jacobcollier/Desktop/Summer_2024/BERT/UNResProcessorPDF/UNResolutionData'
output_file = classify_clauses(file_path)
excel_file_name = f'ACCURACY_UNResolutionData_2024_07_29_0409PM.csv.xlsx' 
excel_file_path = os.path.join(folder_name, excel_file_name)
ToExcel.add_dataframe_to_excel(excel_file_path, 'Final PDF Data', output_file)
print(f"Created new Excel file: {excel_file_path}")
"""

import pandas as pd

# Define the function to classify clauses based on the conditions
def classify_clauses(file_path):
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Ensure that the necessary columns exist
    if 'clause' not in df.columns:
        raise ValueError("The Excel file must have a 'clause' column")
    
    # Function to check if clause begins with a capital letter and ends with a period, semicolon, or comma
    def check_clause(clause):
        if clause and isinstance(clause, str):
            return 1 if clause[0].isupper() and clause[-1] in ['.', ';', ','] else 0
        return 0
    
    # Apply the function to each clause and store the result in the 'correct' column
    df['correct'] = df['clause'].apply(check_clause)
    
    # Create a new Excel file name by appending '_classified' to the original file name
    output_file = file_path.replace('.xlsx', '_classified.xlsx')
    
    # Save the modified DataFrame back to a new Excel file
    df.to_excel(output_file, index=False)
    
    return output_file

# Example usage:
file_path = '/Users/jacobcollier/Desktop/Summer_2024/BERT/UNResProcessorPDF/UNResolutionData/annotated_UNResolutionData_2024_08_05_1147PM copy.xlsx'
output_file = classify_clauses(file_path)
print(f"Classified file saved to: {output_file}")