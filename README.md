# Machine Learning  & Natural Language Processing - Using Naïve Bays Classifier.

- In this program I use a dataset of Hacker News fetched form kaggle to build a probabilistic model from the training set. The code parse the
files in the training set and builds a vocabulary with all the words it contains in Title which is Created
At 2018. Then for each word, I compute their frequencies and the probabilities of each Post Type
class (story, ask_hn, show_hn and poll). 
- After that I extract the data from Created At 2019 as the testing dataset and using a Naïve Bays classifier I predict the class type of each title in the testing set and compare it to the orginal one.
- Finally, I conduct 3 different experiements to play with the classifier in order to see how it will improve its performance.
I generate two graphs as shown below to plot both performance of the classifiers against the number of words left in your vocabulary.

<img width="911" alt="Screen Shot 2020-06-27 at 12 27 54 PM" src="https://user-images.githubusercontent.com/35696935/85927064-a8bd1280-b871-11ea-968e-16344595daef.png">

<img width="512" alt="Screen Shot 2020-06-27 at 12 28 45 PM" src="https://user-images.githubusercontent.com/35696935/85927080-c5f1e100-b871-11ea-8625-62b3235357ec.png">

## Getting Started
### Prerequisites

List of libraries to install:

```
re
pandas
nested_dict
math
sklearn.metrics
matplotlib

```

## Running the program

Instruction on how to play the program:
- Just press the run button and everything will be generated for you.
- If you wish to test my code on a new csv with the same classes just specify the file_name in line 615.
- If you want to run specific part of the assignment you will need to comment out some parts of the code.

## Authors

* **Robert Beaudenon**


