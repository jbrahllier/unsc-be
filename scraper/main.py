"""
UNResolutionProcessor: main.py

    RUN THIS FILE
        NOTE: this project is setup for use with macOS and Firefox
        NOTE: ON SELENIUM (web scraping new resolutions) - if you're scraping new resolutions for the first time, you will likely be 
              prompted to enter your password; enter your password, then kill the terminal and re-run main.py if issues persist on the
              first run

Modules:
    - GrabResID: grabs the res id from a pdf file
    - PDFchunker: splits processed text (with italic markings) into quasi-sentences
    - PDFextractor: processes res pdf files, reading them to text
    - ProcessHelpers: some basic input/animation/time/etc helper functions
    - ReadAndProcessPDFs: processes all the res pdf files in a directory and outputs the dataframe
    - ResFisher: uses selenium to scrape res pdf files from the UN website, storing them in a directory
    - ResolutionSlimmer: some basic text cleaning functions
    - ToExcel: exports processed data to Excel
    - clauseContextualizer: preceding and following quasi-sentences are concatenated around each quasi-sentence, new column
    - createSubsetToAnnotate: makes new data randomized around the quasi-sentences, sampled at some percent of the total data
    - duplicateRemover: removes any quasi-sentence dups
    - phraseRemover: helpful for removing redundant quasi-sentences

System-Level Dependencies:
    - LibreOffice
    - unoconv

Other Dependencies
    - pandas, re, os, itertools, time, pdfplumber, pymupdf, fitz, string, unidecode, docx2pdf,
      pickle, openpyxl, requests, bs4, selenium, datetime, logging, io, contextlib

Usage:
    Run the script in a Python environment with internet access to install missing packages, respond to prompts, 
    wait, enjoy data
"""
# Developer Mode: used to bypass user-input prompting
#   NOTE: if 'True', fill out lines 59-68 with your selections (or run the default)
dev_mode = False

def main():
    # installation of system-level and other dependencies for the whole res processor
    import installer
    installer.run_install_script()

    # import modules and other dependencies
    import ProcessHelpers
    from ProcessHelpers import animated_printer
    import os
    import time 
    from datetime import datetime
    from ToExcel import add_dataframe_to_excel 
    import ReadAndProcessPDFs
    import ResFisher
    import duplicateRemover
    import createSubsetToAnnotate
    import clauseContextualizer
    import duplicateRemover
    import phraseRemover

    animated_printer.safe_print("Imports complete. Running the program...")

    if dev_mode:
        pullNewPDFs = False # pull new res toggle
        folder_path_input = 'UNPDFs' # insert pdf data path here
        folder_path = os.path.join(os.path.dirname(__file__), folder_path_input)
        exclude_annex = True # exclude annex toggle
        remove_duplicates = True # remove duplicates toggle
        strings_to_remove = ['The Security Council,', 'Decides to remain seized of the matter.', 
                            'Decides to remain actively seized of the matter.'] # fill out list of phrases to remove
        remove_phrases = True # remove phrases toggle
        create_subset_for_annotation = True # create annotation data toggle
        contextualize_for_annotation = True # concatenate clause context toggle
        animated_printer.safe_print("Developer Mode activated...")
    else: 
        # if true, this prompts the use of selenium to scrape new resolutions from the UN website
        pullNewPDFs = ProcessHelpers.get_user_input("Do you want to pull new UNSC Resolutions from the UN website?")

        # specifies res pdf directory
        if pullNewPDFs:
            folder_path_input = input("Enter the name of the folder you'd like to store the resolutions in (NOTE: if the folder doesn't exist or isn't found in this script's directory, one will be made for you):").strip()
            folder_path = os.path.join(os.path.dirname(__file__), folder_path_input)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        else: 
            folder_path_input = input("Enter the name of the folder with your UNSC Resolutions (NOTE: just the folder, not the whole path): ").strip()
            while True:
                folder_path = os.path.join(os.path.dirname(__file__), folder_path_input)
                if os.path.exists(folder_path) and os.path.isdir(folder_path):
                    continue_creation = ProcessHelpers.get_user_input(f"Are you sure you would like to use the folder '{folder_path}'?")
                    if continue_creation:
                        break
                    else: 
                        folder_path_input = input("Enter the name of the folder you would like to use: ").strip()
                else:
                    folder_path_input = input("Folder does not exist. Please enter a valid folder path: ").strip()
        animated_printer.safe_print(f"Using folder: {folder_path}")

        # exclude annex?
        exclude_annex = ProcessHelpers.get_user_input("This program EXCLUDES the processing of resolution annexes. Is that OK? (answer 'N' to INCLUDE annexes)")
        
        # remove dups?
        remove_duplicates = ProcessHelpers.get_user_input("Would you like to remove duplicates in the data?")
        
        # remove phrases?
        strings_to_remove = ['The Security Council,', 'Decides to remain seized of the matter.', 
                            'Decides to remain actively seized of the matter.'] # specify phrases
        remove_phrases = ProcessHelpers.get_user_input("Would you like to remove select phrases in the data?")

        # create subset for annotation/verification? contextualize quasi-sentences?
        create_subset_for_annotation = ProcessHelpers.get_user_input("Would you like to create a subset for annotation? (this reshuffles the data at random, then removes all but a specified percentage)")
        if create_subset_for_annotation: 
            while True:
                try:
                    annotation_percentage = int(input('What percentage of the data would you like to annotate? (between 5-15 percent is recommended) [enter an int between 0-100]: '))
                    if 0 <= annotation_percentage <= 100:
                        print(f"Will annotate {annotation_percentage}% of the data.")
                        break
                    else:
                        print("Error: Percentage must be between 0 and 100.")
                except ValueError:
                    print("Error: Input must be an integer.")
        contextualize_for_annotation = ProcessHelpers.get_user_input("Would you like to contextualize the clauses for annotation? (this concatenates the preceding and following clauses to the clause to contextualize annotation, adding it to a new column)")


    # recording program runtime
    start_time = time.time() 

    # scraping resolutions from the UN website (if true)
    if pullNewPDFs:
        animated_printer.safe_print("This could take a while! Leave your computer on, open, and connected to Wi-Fi.")
        animated_printer.animate(True) 
        ResFisher.resFisher(folder_path)
        animated_printer.animate(False) 
        animated_printer.safe_print("Completed scraping the resolutions from the UN website. Creating new data now...")
    else:
        animated_printer.safe_print("This could take a while! Leave your computer on, open, and connected to Wi-Fi.")
        animated_printer.safe_print("Creating new data now...")

    # process resolution pdfs, output dataframe
    animated_printer.animate(True) 
    failed_pdfs = []
    annex_pdfs = []
    final_data = ReadAndProcessPDFs.process_all_files(folder_path, failed_pdfs, exclude_annex, annex_pdfs)
    animated_printer.animate(False) 
    animated_printer.safe_print("Finished creating data.")

    # removing duplicates from dataframe (if true)
    if remove_duplicates:
        animated_printer.safe_print("Removing duplicates...")
        final_data = duplicateRemover.remove_duplicates_and_adjust_ids(final_data, 'clause', 'clauseID')

    # removing phrases from dataframe (if true)
    if remove_phrases:
        animated_printer.safe_print("Removing phrases...")
        final_data = phraseRemover.remove_strings_and_adjust_ids(final_data, 'clause', strings_to_remove, 'clauseID')

    # removing phrases from dataframe (if true)
    if contextualize_for_annotation:
        animated_printer.safe_print("Adding context column...")
        final_data = clauseContextualizer.concatenate_quasi_sentences(final_data)

    # saving dataframe as timestamped .csv and .xlsx files
    folder_name = 'UNResolutionData' # i've decided on this folder name, feel free to change it
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # get timestamp 
    now = datetime.now() 
    date_str = now.strftime('%Y_%m_%d_%I%M%p')

    # save dataframe to .csv
    csv_file_name = f'UNResolutionData_{date_str}.csv'
    csv_file_path = os.path.join(folder_name, csv_file_name)
    final_data.to_csv(csv_file_path, index=False)
    animated_printer.safe_print(f"Created new CSV file: {csv_file_path}")

    # save dataframe to .xlsx
    excel_file_name = f'UNResolutionData_{date_str}.xlsx' 
    excel_file_path = os.path.join(folder_name, excel_file_name)
    add_dataframe_to_excel(excel_file_path, 'Final PDF Data', final_data)
    animated_printer.safe_print(f"Created new Excel file: {excel_file_path}")

    # save subset dataframe to .csv (open in excel to annotate)
    if create_subset_for_annotation:
        animated_printer.safe_print("Creating subset for annotation...")
        createSubsetToAnnotate.createSubsetToAnnotate(csv_file_path, annotation_percentage)
        animated_printer.safe_print(f"Created a subset for annotation at {csv_file_path}.")

    # recording program runtime
    end_time = time.time() 
    elapsed_time = end_time - start_time 

    # list failed pdfs and print num failed pdfs
    if len(failed_pdfs) > 0:
        animated_printer.safe_print("Some resolutions failed to process. ")
        animated_printer.safe_print("List of failed PDFs: " + str(failed_pdfs))
        animated_printer.safe_print("Number of failed PDFs: " + str(len(failed_pdfs)))
    else:
        animated_printer.safe_print("All resolutions properly processed.")

    # print num pdfs with annexes
    if (len(annex_pdfs) > 0) and exclude_annex:
        # NOTE: include the below to print out the list of resolutions with annexes
        # ProcessHelpers.safe_print("List of resolutions with annexes processed: " + str(annex_pdfs))
        animated_printer.safe_print("Number of resolutions with annexes (annexes removed): " + str(len(annex_pdfs)))
    elif (len(annex_pdfs) == 0) and exclude_annex:
        animated_printer.safe_print("No resolutions processed had annexes.")

    # processing conlcuded output
    animated_printer.safe_print(f"The script took {ProcessHelpers.format_elapsed_time(elapsed_time)} to run all processes.")
    ProcessHelpers.print_num_files_in_directory(folder_path)
    animated_printer.safe_print("Task complete.")

if __name__ == "__main__":
    main()
