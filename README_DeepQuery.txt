DeepQuery: A Rule-Based Masters’ SOP Evaluator

HOW TO RUN?

## Libraries Required

Before running the notebook, ensure that the following libraries are installed. You can install them directly in your Jupyter Colab notebook using the provided commands:

!pip install spacy nltk textblob language-tool-python transformers textstat matplotlib seaborn scikit-learn
!python -m spacy download en_core_web_sm

List of Required Libraries:
•	spacy
•	nltk
•	textblob
•	language-tool-python
•	transformers
•	textstat
•	matplotlib
•	seaborn
•	scikit-learn

Usage Instructions
Step 1: Input Your SOP
1.	Open the Jupyter Colab Notebook:
	Navigate to your Jupyter Colab environment and open the SOP_Evaluator.ipynb notebook.
2.	Locate the sop_text Variable:
	In the notebook, find the cell where the sop_text variable is defined.
3.	Replace the Placeholder with Your SOP:
	Replace the existing placeholder text with your own Statement of Purpose (SOP) enclosed within triple quotes. 

For example:

sop_text = """
I am [Your Name], teaching assistant at the Mechanical Design and Production department...
# Your full SOP goes here
"""

Step 2: Select Program-Specific Keywords and Skills

1.	Access the Compiled Keyword and Skills Lists:
	I have precompiled extensive lists of program-specific keywords and skills for approximately 128 different master's programs. 	These lists are organized in separate files (program_keywords.json, program_skills.json).

2.	Identify Relevant Lists for Your Program:
	Browse through the compiled files to find the keywords and skills that correspond to your specific master's program.

3.	Copy Relevant Keywords and Skills:
	Once identified, copy the relevant keywords and skills from these files.

4.	Paste into the Notebook Placeholders:
	In the notebook, locate the placeholders for program_keywords and skills_list.
	Replace the placeholder lists with your copied keywords and skills. 

For example:
# Define program-specific keywords
program_keywords = [
    'solid mechanics', 'mechanical engineering', 'finite elements', 'computational modeling',
    # ... add all relevant keywords for your program
]

# Define skills list
skills_list = [
    'MATLAB', 'AutoCAD', 'SolidWorks', 'ANSYS', 'COSMOS Motion', 'Abaqus CAE', 'Python',
    # ... add all relevant skills for your program
]

Step 3: Run the Evaluator
1.	Execute All Cells:
	Run all the cells in the notebook sequentially to perform the evaluation.
2.	Review the Output:
	The evaluator will process your SOP and generate scores across various metrics.
	Visualizations, such as bar charts, will display the evaluation results, highlighting strengths and areas for improvement.


Note:
1. Ensure Accuracy: Make sure that the keywords and skills you select accurately reflect the requirements and focus areas of your desired master's program to obtain meaningful evaluation results.
2. Grammar and Readability: Before running the evaluator, it's advisable to review your SOP for grammatical correctness and readability to ensure optimal scoring.
3. Consistency: While the evaluate_consistency function is currently a placeholder, future updates may include checks for consistency with other application materials.


REGARDING THE DATASET:
https://mahindraecolecentrale-my.sharepoint.com/:f:/g/personal/se22uari048_mahindrauniversity_edu_in/EqRQnl7atT5Akr4EMi3BBUgBAbDhFCHVMLMJb_Buw1n0FA

Folder: 'DATASETS':
Talking about the dataset, there wasn't any ready made dataset available on Kaggle or elsewhere. We've had to scrape the web and manually download data.
The following are the contents of the folder.

categorized: This subfolder contains a total of 1400 SOP samples, which are categorized.
The 1400 samples span the following branches/fields of study:
1. Computer Science (300)
2. Biotechnology (100)
3. Aeronautical Engineering (100)
4. Archaeology (100)
5. Civil Engineering (100)
6. Electrical Engineering (100)
7. History (100)
8. Journalism (100)
9. Marine Engineering (100)
10. MBA (100)
11. Mechanical Engineering (100)
12. Philosophy (100)

uncategorized: This subfolder contains 123 SOP samples that aren't categorized. These samples also span various fields and levels of study.


Regarding the program_keywords & program_skills json files:

Defined them in json files so that if at all you want to use the 'program:keyword' association pairs in some other project or elsewhere,
you can just download the json file and use it because we've defined everything in a 'key:value' format.

Regarding the File: 'Breaking down an SOP.pdf':
This file fully discusses in detail what an SOP (Statement of Purpose) is and answers the following questions:
1. What is an SOP (Statement of Purpose)?
2. Why is an SOP written?
3. How should an SOP be written?
4. What are the components of an SOP?
5. What makes one SOP good and another one bad?
6. What should an SOP convey?
7. Additional Important Information.

We've spent many a day researching and compiling the above information, as there is no point in rushing things up without properly knowing what an SOP is in the first place. We also believe this is what will form the backbone when judging & providing feedback on SOP's. Also, having this information is crucial to generate a custom score for every SOP.

We plan to evaluate SOP's that user provides based on all the metrics listed in the file before generating a custom score. Refer the documentation & Analysis file to know more about the metrics/rubrics.
