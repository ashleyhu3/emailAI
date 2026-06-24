from pipeline import PDFSummarizerPipeline
import os
from dotenv import load_dotenv
load_dotenv()  


DB_URL = os.getenv("PDF_SUMMARIZER_DB_URL")
pipeline = PDFSummarizerPipeline(database_url=DB_URL)

pdf_file = "/Users/davidfu/Desktop/Rays_Intern/PDF_summarizer/research_pdfs/1_91APP (6741) - HK Roadshow - Nov 7-8th, 2024_91App HK Road Nov 7-8.pdf"

result = pipeline.process_single_pdf(pdf_file)
print(result)

#intial problem: have an AI so powerful = understand anything, unlimited context
#break down data = shrink context of datat, easier to understand

#is there a way to shrink the data --> easier for the AI to understand

#individually summarize each PDF

#document chunking: more sense for sectioning to be done by AI?

#first implement double summary just to start with
#AI vectorizes PDFs
#   -who report is from, what type of report is it, some structure to how we break it down
#ideal: Gmail w/ research pdf. Put into database w/ author, general summary, broker, etc.
# once we put the PDFs together, we try and structure it more

#step 1) each pdf summarized + various classifications. Is it even better to be separated into chunks? Lose context, especially with longer PDFS
#   try different prompts of how the AI breaks down various PDFs
#   Saved as raw text
#   Once pdf in database, separated micro-scale

#focus on the PDF Summarization first


#testing what is more reliable? AI read report, or using metadata in the database + query

#larger context you feed to AI, context a) not properly understood + b) hallucinates more, w/ more data you give
# increase the consistency where AI fully absorbs all relavant data, reduce chance of hallucination, + prompt in a more structured manner
#pulls data from actually relavant reports
#ask for a summary of an industry
#start with a few classifications first --> move forward from there
#can tell the user you can't prompt certain things

#need 99% accuracy, low tolerance for mistakes
#in the future, add some confidence interval
#future, summarize transcripts + audio data

#Work towards Demo in March
# 50 PDFs, already have a summary in the database
#see what original vs summary looks like + classifications it can identify
#step 2) AI based on a simple prompt of specifications --> pull out those specific summaries 
# step 3) what do you want from these 10 research reports (last AI uses the summarized data to answer the question)
# Big question: At what point is the data too summarized to where the data is lost

#play around with the meta data
#if we can demo this, then move on to expanding it

#if there is any doubts about the accuracy of information, it is basically worthless

#biggest problems are reliability and accuracy

# demo: 2 prompts, general prompt vs more specific prompt

#research reports: see the before an after
#classification specification
#did it miss any, did it get all of them