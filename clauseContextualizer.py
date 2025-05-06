"""
UNResolutionProcessor: clauseContextualizer.py

    concatenates the former and successive quasi-sentences to each quasi-sentence -- contextualizes the quasi-sentence for
    the annotator (resets by res, handles bounds by taking the current + successive and former + current, respectively)

"""
"""
import pandas as pd

def concatenate_quasi_sentences(df, clause_col="clause", clause_id_col="clauseID", doc_id_col="resID", new_col_name="context_clause"):
    # sort dataframe by res ID and clause ID
    df = df.sort_values(by=[doc_id_col, clause_id_col])
    
    # new column
    df[new_col_name] = ''
    
    # process on res by res basis
    for doc_id in df[doc_id_col].unique():
        doc_df = df[df[doc_id_col] == doc_id]
        for idx in doc_df.index:
            if idx == doc_df.index[0]:  # beginning res boundary
                concatenated = doc_df.loc[idx, clause_col] + ' ' + doc_df.loc[idx + 1, clause_col]
            elif idx == doc_df.index[-1]:  # ending res boundary
                concatenated = doc_df.loc[idx - 1, clause_col] + ' ' + doc_df.loc[idx, clause_col]
            else:  # meat and potatoes
                concatenated = (doc_df.loc[idx - 1, clause_col] + ' ' + 
                                doc_df.loc[idx, clause_col] + ' ' + 
                                doc_df.loc[idx + 1, clause_col])
            
            # trim fat and add to dataframe
            df.at[idx, new_col_name] = concatenated.strip()
    
    return df
"""
import pandas as pd

def concatenate_quasi_sentences(df, clause_col="clause", clause_id_col="clauseID", doc_id_col="resID", new_col_name="context_clause"):
    # sort dataframe by res ID and clause ID
    df = df.sort_values(by=[doc_id_col, clause_id_col])
    
    # new column
    df[new_col_name] = ''
    
    # process on res by res basis
    for doc_id in df[doc_id_col].unique():
        doc_df = df[df[doc_id_col] == doc_id]
        doc_index = doc_df.index.tolist()
        for i, idx in enumerate(doc_index):
            if i == 0:  # beginning res boundary
                if i + 1 < len(doc_index):  # check if there is a next clause
                    concatenated = doc_df.loc[idx, clause_col] + ' ' + doc_df.loc[doc_index[i + 1], clause_col]
                else:
                    concatenated = doc_df.loc[idx, clause_col]
            elif i == len(doc_index) - 1:  # ending res boundary
                if i - 1 >= 0:  # check if there is a previous clause
                    concatenated = doc_df.loc[doc_index[i - 1], clause_col] + ' ' + doc_df.loc[idx, clause_col]
                else:
                    concatenated = doc_df.loc[idx, clause_col]
            else:  # meat and potatoes
                concatenated = (doc_df.loc[doc_index[i - 1], clause_col] + ' ' + 
                                doc_df.loc[idx, clause_col] + ' ' + 
                                doc_df.loc[doc_index[i + 1], clause_col])
            
            # trim fat and add to dataframe
            df.at[idx, new_col_name] = concatenated.strip()
    
    return df