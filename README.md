# acronym-generator
Generate smart acronyms from a given expression!  

Run generator.py to generate smart acronyms   
-set maximum and minimum acronym length, and letters used per word in the expression  
-use synonyms or a variable word order  

Dependencies:   
  numpy  
  cvxpy  
  Flair  
  nltk  
  gensim  
  pronounceable   
  
  Also requires a wikipedia-scraped dataset of wordfrequencies, like the enwiki-20190320-words-frequency.txt from https://github.com/IlyaSemenov/wikipedia-word-frequency  
 
Creates a vectors.kv file on first use if this file is not cloned   
  
  These last two files are too big to upload, the wordfrequencies dataset has to be downloaded manually from the above repo
