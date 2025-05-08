"""
UNResolutionProcessor: ResFisher.py

    uses selenium/soup to scrape res pdf files from the UN website, storing them in a directory
    # NOTE: LibreOffice dependency
    # NOTE: unoconv dependency
    # NOTE: setup to work with Firefox 

"""
import os
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from datetime import datetime
import mimetypes
from ProcessHelpers import animated_printer
import time
from selenium.common.exceptions import WebDriverException, TimeoutException

def resFisher(save_directory):
    # set up selenium with Firefox
    options = Options()
    options.headless = True

    # set Firefox preferences to handle downloads automatically
    profile = webdriver.FirefoxProfile()
    profile.set_preference("browser.download.folderList", 2)
    profile.set_preference("browser.download.manager.showWhenStarting", False)
    profile.set_preference("browser.download.dir", save_directory)
    profile.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/pdf, application/msword")
    options.profile = profile
    driver = webdriver.Firefox(options=options)

    while True:
        try:
            driver = webdriver.Firefox(options=options)
            break
        except WebDriverException as e:
            print("Please ensure Firefox has the necessary permissions (kill the terminal and re-run main.py if this message continues to appear or you run into other issues).")
            input("Press Enter after granting permissions and closing any dialog boxes...")
            # retry after the user has granted permissions

    # define base URL
    base_url = "https://www.un.org/securitycouncil/content/resolutions-adopted-security-council-"

    # check/define save directory
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    def download_file(file_url, save_path):
        # download and save a pdf (or word) document
        try:
            response = requests.get(file_url)
            response.raise_for_status()
            with open(save_path, 'wb') as file:
                file.write(response.content)
        except Exception as e:
            animated_printer.safe_print(f"Failed to download {file_url}: {e}")

    def get_custom_filename(resolution_string):
        # pull desired filename from the download link text on the webpage
        return resolution_string.replace("/", "_") + ".pdf"

    def convert_doc_to_pdf(doc_path, pdf_path):
        # convert doc to pdf using unoconv
        try:
            os.system(f'unoconv -f pdf -o "{pdf_path}" "{doc_path}"')
            os.remove(doc_path)  # remove the doc file after conversion
            animated_printer.safe_print(f"Converted and saved: {pdf_path}")
        except Exception as e:
            animated_printer.safe_print(f"Failed to convert {doc_path} to PDF: {e}")

    # iterate through the years from 2024 to 2000 (stopping after S/RES/1293 in 2000)
    for year in range(datetime.now().year, 1999, -1):
        year_url = f"{base_url}{year}"
        driver.get(year_url)

        WebDriverWait(driver, 10).until( # wait to load
            EC.presence_of_element_located((By.TAG_NAME, 'body'))
        )

        year_soup = BeautifulSoup(driver.page_source, 'html.parser')

        # find links with the pattern 'S/RES/...'
        res_links = [a['href'] for a in year_soup.find_all('a', href=True) if 'S/RES/' in a['href']]
        break_out = False
        for res_link in res_links:
            resolution_string = res_link.split("/")[-1]  # extract 'S/RES/...'
            if int(resolution_string[:4]) == 1292:  # break out of the loop at S/RES/1292 (this is the last '90s-style' resolution the UN released in 2000)
                animated_printer.safe_print("Resolution scraping complete.")
                break_out = True
                break
            res_page_url = res_link if res_link.startswith("http") else f"https://www.un.org{res_link}"
            driver.get(res_page_url)

            WebDriverWait(driver, 10).until(  # wait to load
                EC.presence_of_element_located((By.TAG_NAME, 'body'))
            )

            res_page_soup = BeautifulSoup(driver.page_source, 'html.parser')

            # check for a language selection page
            english_link = res_page_soup.find('a', href=True, string='English')
            if not english_link:
                english_link = res_page_soup.find('a', href=True, text=lambda t: t and 'Lang=E' in t)

            if english_link:
                pdf_page_url = english_link['href'] if english_link['href'].startswith("http") else f"https://www.un.org{english_link['href']}"

                # track files before download
                before_files = set(os.listdir(save_directory))

                # open the link to trigger download
                try:
                    driver.get(pdf_page_url)
                except TimeoutException:
                    animated_printer.safe_print(f"Timeout while trying to access {pdf_page_url}. Checking for downloads.") # Need to get rid of this

                # wait for download to complete
                time.sleep(10) 

                # track files after download
                after_files = set(os.listdir(save_directory))
                new_files = after_files - before_files

                # check for downloaded .doc file
                doc_files = [filename for filename in new_files if filename.endswith('.doc')]
                if doc_files:
                    for doc_file in doc_files:
                        doc_path = os.path.join(save_directory, doc_file)
                        pdf_path = os.path.join(save_directory, get_custom_filename(resolution_string))
                        convert_doc_to_pdf(doc_path, pdf_path) # if so, convert to pdf
                    continue 

                file_url = pdf_page_url
            else:
                # if no pdf page/english pdf page link, assume the link itself is the download link
                file_url = res_page_url

            # get the directory right
            custom_filename = get_custom_filename(resolution_string)
            save_path = os.path.join(save_directory, custom_filename)

            # check if the file already exists
            if os.path.exists(save_path):
                animated_printer.safe_print("Your folder is now up to date with the most recent UNSC Resolutions.")
                break_out = True
                break

            download_file(file_url, save_path)

            # check MIME type and convert if necessary (double-check/if not an english link)
            mime_type, _ = mimetypes.guess_type(save_path)
            if mime_type == 'application/msword':
                doc_path = save_path
                pdf_path = save_path.replace(".doc", ".pdf")
                convert_doc_to_pdf(doc_path, pdf_path)
                animated_printer.safe_print(f"Converted and saved: {pdf_path}")

        if break_out:
            break

    # close the browser
    driver.quit()
