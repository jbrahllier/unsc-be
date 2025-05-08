"""
UNResolutionProcessor: PDFextractor.py

    processes res pdf files, reading them to text -- processes outlined below:
    1. extracts the date from the res pdf file
    2. removes text in headers and footers, whittling clean the main boxes of text for processing (might also remove annexes)
        - uses the position of 'Resolution' in bold at the top of the page as a reference point for anchoring the main
          body of text (gives a cushiony buffer, needed for less-than-tidy resolutions, of which there are no more than five)
        - detects content by drawing lines, getting cut lines based on the position of content it finds to the left and 
          right of the main-body area
        - [might remove everything after the annex]
        - creates clean content boxes used to extract the text from the main res body
    3. Wraps valid italic phrases in text-based indicators for chunking

"""
import pdfplumber
import fitz  # or 'PyMuPDF'
import re
import GrabResID
from ProcessHelpers import animated_printer

def extract_date_from_pdf(pdf_path):
    # hides recoverable MuPDF error messages to ignore unnecessary scaries
    fitz.TOOLS.mupdf_display_errors(False)

    # open up
    document = fitz.open(pdf_path)
    if not document:
        animated_printer.safe_print("Could not open the PDF file.")
        return 0, '', 0  # return empty if the file cannot be opened (check final .csv / .xlsx for verification/troubleshooting)
    
    # get first page
    first_page = document[0]
    
    # extract its text
    text = first_page.get_text()
    
    # find the first date at the top
    date_pattern = r'\d{1,2} [A-Za-z]+ \d{4}'
    match = re.search(date_pattern, text)

    if match:
        date_str = match.group(0)
        # extract day, month, and year from the date string
        day_str, month, year_str = date_str.split()
        day = int(day_str)
        year = int(year_str)
        return day, month, year
    else:
        animated_printer.safe_print(f"Date not found in the PDF of Resolution {GrabResID.grab_resID(pdf_path)}.")
        return 0, '', 0  # see note on this above

def find_resolution_position(page):
    # find the position of 'Resolution' (from the left bound of the 'R')
    for word in page.extract_words():
        if word['text'] == 'Resolution':
            return word['x0'], word['top']
    return None, None

def get_cut_lines(page, resolution_x=None, resolution_y=None):
    page_height = int(page.height)  # convert to int
    page_width = int(page.width)  # get page width
    horizontal_center_y = int(page_height / 2)  # convert to int
    
    if resolution_x is None or resolution_y is None:
        # default values if "Resolution" is not found (it's always found)
        # NOTE: but if it isn't (it is), these values are always inclusive of relevant clause data
        header_line_y = 15
        footer_line_y = page_height - 15
    else:
        # vertical boundary lines
        vertical_line_x_left = resolution_x - 30 # NOTE: changed this from 2 to 30 (bigger puff needed)
        vertical_line_x_right = page_width - resolution_x + 30 # NOTE: same as above
        
        # find header line (looking up)
        header_line_y = 0
        footer_line_y = page_height
        for y in range(horizontal_center_y, 0, -1):
            # check for content to the left of the first line or the right of the second line
            if detect_any_content(page, y, vertical_line_x_left) or detect_any_content(page, y, vertical_line_x_right, right=True):
                header_line_y = y + 1
                break
        
        # find footer line (looking down)
        for y in range(horizontal_center_y, page_height):
            # same as finding the header line
            if detect_any_content(page, y, vertical_line_x_left) or detect_any_content(page, y, vertical_line_x_right, right=True):
                footer_line_y = y - 1
                break
    
    return header_line_y, footer_line_y

def detect_any_content(page, y, vertical_line, right=False):
    if right: # check to the right of the line
        # text content
        if page.crop((vertical_line, y, page.width, y + 1)).extract_text().strip():
            return True
        # line content
        for line in page.lines:
            if line['top'] <= y <= line['bottom'] and line['x0'] >= vertical_line:
                return True
        # image content
        for image in page.images:
            if image['top'] <= y <= image['bottom'] and image['x1'] >= vertical_line:
                return True
        # rect content
        for rect in page.rects:
            if rect['top'] <= y <= rect['bottom'] and rect['x1'] >= vertical_line:
                return True
    else: # check to the left of the line
        # text content
        if page.crop((0, y, vertical_line, y + 1)).extract_text().strip():
            return True
        # line content
        for line in page.lines:
            if line['top'] <= y <= line['bottom'] and line['x1'] <= vertical_line:
                return True
        # image content
        for image in page.images:
            if image['top'] <= y <= image['bottom'] and image['x0'] <= vertical_line:
                return True
        # rect content
        for rect in page.rects:
            if rect['top'] <= y <= rect['bottom'] and rect['x0'] <= vertical_line:
                return True
    return False

def get_main_content_bboxes(file_path, exclude_annex):
    annex_pdf = None
    if not exclude_annex: # annex inclusive
        bboxes = []
        with pdfplumber.open(file_path) as pdf:
            resolution_x, resolution_y = None, None
            for i, page in enumerate(pdf.pages): # iterate through all pdf pages
                if i == 0:
                    resolution_x, resolution_y = find_resolution_position(page)
                header_line_y, footer_line_y = get_cut_lines(page, resolution_x, resolution_y) # get cut lines
                bbox = (0, header_line_y, page.width, footer_line_y) # define bbox 
                bboxes.append(bbox) # add bbox to list
        return bboxes
    else: # annex exclusive (notes same as above, just exclude all bboxes after the annex)
        annex_page = search_annex_bold_in_all_pages(file_path)
        if annex_page:
            annex_pdf = file_path
        bboxes = []
        with pdfplumber.open(file_path) as pdf:
            resolution_x, resolution_y = None, None
            for i, page in enumerate(pdf.pages):
                if annex_page and (i + 1) >= annex_page:
                    break
                if i == 0:
                    resolution_x, resolution_y = find_resolution_position(page)
                header_line_y, footer_line_y = get_cut_lines(page, resolution_x, resolution_y)
                bbox = (0, header_line_y, page.width, footer_line_y)
                bboxes.append(bbox)
        return bboxes, annex_pdf

def contains_annex(page):
    # just looks at the top of the page
    crop_height = page.height * 0.2
    cropped_page = page.within_bbox((0, 0, page.width, crop_height))
    
    # only grab it if it's bold (this + the crop makes sure it only grabs the mention of 'Annex' marking the annex)
    bold_text = cropped_page.extract_words(extra_attrs=["fontname"])
    for word in bold_text:
        if word['text'] == 'Annex' and 'Bold' in word['fontname']:
            return True
    return False

def search_annex_bold_in_all_pages(pdf_path):
    annex_page = None
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            if contains_annex(page): # marks the page where the annex begins, so elsewhere the rest of the res can be ignored
                annex_page = i + 1
                break
    return annex_page

def extract_text_with_italics(file_path, bboxes): # NOTE: this was buggy to build, so it comes with some error testing residue
    # [message copied from above] hides recoverable MuPDF error messages to ignore unnecessary scaries
    fitz.TOOLS.mupdf_display_errors(False)

    # error detection for fitz problems
    try:
        doc = fitz.open(file_path)
    except fitz.fitz.FileDataError as e:
        animated_printer.safe_print(f"FileDataError: {e}")
        return ""
    except fitz.fitz.MuPdfError as e:
        animated_printer.safe_print(f"MuPdfError: {e}")
        return ""
    except Exception as e:
        animated_printer.safe_print(f"An unexpected error occurred while opening the file: {e}")
        return ""

    formatted_text = ""
    try:
        for i, page in enumerate(doc):
            if i >= len(bboxes):
                # skips if there are no bounding boxes on a page (there never aren't, just a precaution)
                continue

            # get bbox
            bbox = bboxes[i]
            try:
                text_page = page.get_text("dict", clip=fitz.Rect(bbox))
            except Exception as e:
                animated_printer.safe_print(f"An error occurred while extracting text from page {i+1}: {e}")
                continue

            # wrap italicized text in signatures to preserve italics in plain text
            for block in text_page.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"]
                            try:
                                # check if the text is italic and begins with an upper case letter
                                if span["flags"] & 2 and text[0].isupper():  
                                    formatted_text += f"<i>{text}</i>"
                                else:
                                    formatted_text += text
                            except Exception as e:
                                animated_printer.safe_print(f"An error occurred while processing text span: {e}")
                        formatted_text += " "
    except Exception as e:
        animated_printer.safe_print(f"An unexpected error occurred while processing the document: {e}")

    return formatted_text