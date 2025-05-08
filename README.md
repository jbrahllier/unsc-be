# unsc-bert
This repository contains scripts developed to webscrape UNSC Resolutions, tune BERT classifiers (actor + sentiment), and predict sentiment towards different actors in the Security Council.

These scripts were developed between June 2024 and May 2025. 

The 'scraper' folder contains every file needed to scrape fresh UNSC Resolutions from the UN website (NOTE: the scraper may need adjustment if the UN website has changed substantially since October 2024).

The 'models' folder contains every file needed to fine-tune a BERT-based, multi-label classification model (specifically the one from Will Horne), run inference using a fine-tuned model either saved to a local machine or uploaded to HuggingFace, and run zero-shot sentiment classification on each of the actors using BERT-NLI (Natural Language Inference) methodology 

One thing to note is that there is currently *no* support for fine-tuning the sentiment model, though this should not be difficult to do; Amy and Susan have plenty of labeled data that, when regularized to the set of actors on the actor classification head, may be used to fine-tune a sentiment model. That means more data to train on, and that a simple "Premise: '______'. Hypothesis: 'The sentiment towards {actor} is negative/positive/neutral'." can be trained on data with only a single actor labeled (which the old labeled data is) and then *applied* to the multi-actor pipeline.

## Using The Scraper

All you should need to do to run the scraper is run 'scraper/main'. The script is designed to install any system- or script-level dependencies, then walk the user through the scraping process.

If you run into problems, there are many sub-scripts, all of which, if you go either to 'scraper/main' or the sub-script file, are accounted for in the documentation. Since these scripts are documented with doc-level overviews and in-line comments, I'll spare an extensive explanation and just provide an overview on how the scraper works (plus a couple caviats if things aren't working).

### Overview
1. When you run main, the first step is to install system-level dependencies (brew, unoconv, LibreOffice) and then install the general requirements (the python libraries in requirements.txt). 
2. Next, the script guides the user through the kind of search they want (data range, new data, preferences for cleaning, etc).
3. Then, the script runs the scrape. Keep the computer on and running. This should take anywhere from 1-4 hours, depending on the number of resolutions being scraped, but this number could thin or bloat. (NOTE: the user should have space available for many pdfs and docx files; the scraper depends on downloading those, then accessing them to extract and clean clause data).
4. Finally, you should have access to identical csv and excel files with the clause data organized by Resolution ID, Clause ID, Date, and other relevant metadata.

### Potential Hiccups
- There might be a problem with brew: in that case, seek documentation elsewhere to get it installed
- The UN might have updated its website:
    - If they changed the overarching directory location but the process is the same, just change the links to direct the scraper to the right place.
    - If they changed the way users access the file system, adjust the logic of the scraper to navigate to download pages by inspecting the website using DevTools.
    - If they changed downloading permissions, then that's a shame; the scraper is no longer usable (this is the least likely case). 

## The Model

As specified above, the model scripts have two functions:
1. Multi-Label Classification of Actors
    - This is done by adapting a BERT model for social media actor classification, 'rwillh11/mdeberta_groups_2.0', to detect UNSC Resolution specific actors.
2. Sentiment Analysis Using NLI Methods
    - This is done by using (or adapting and using) a BERT model for general (or UN-tuned) sentiment detection, specifically 'MoritzLaurer/ModernBERT-large-zeroshot-v2.0'.

