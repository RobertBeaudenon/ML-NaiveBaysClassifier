# -------------------------------------------------------
# Written by Robert Beaudenon
# For COMP 472 – Summer 2020
# --------------------------------------------------------

import re
import pandas as pd
import nested_dict as nd
import math
import sklearn.metrics
from matplotlib import pyplot as plt
import matplotlib.lines as mlines

"""

    Transform the column 'Title' of dataframe into a list
    :param df

"""


def df_title_to_list(df):
    return df.Title.tolist()


"""

    Splits the title into a list of words according to the regex rules specified
    A list of characters avoided can be found in the remove_word.txt file
    Those characters are not removed from the strings if they are in the middle of the word
    :param title

"""


def regex_filtering(title):
    return [word.strip('.,"') for word in re.split(r'\s', re.sub(r"(?<!\S)[^\w $-]+|[^\w $]+(?!\S)", '', title))]


"""

    Creates the vocabulary from the list of titles provided 
    returns each word of the vocabulary along with its frequency in a dictionary
    :param list_of_title
    :param post_type : represents the class of the label
    :param words_dict

"""


def title_to_vocab(list_of_title, post_type, words_dict):
    frequency_of_words = 0

    for title in list_of_title:
        words = regex_filtering(title)
        for word in words:
            if word != '':
                frequency_of_words += 1
                word = word.lower()
                if (word in words_dict.keys()) & (post_type in words_dict[word].keys()):
                    words_dict[word][post_type] = words_dict[word][post_type] + 1
                else:
                    words_dict[word][post_type] = 1

    return words_dict


"""

    If certain words in the vocabulary dictionary are in certain classes (post types) and not other this will 
    lead to missing entries in the dict causing calculation errors.
    To solve that we add the class type with a value of 0.
    :param words_dict
    :param post_type : class

"""


def fill_non_existing_values(words_dict, post_type):
    for word in words_dict:
        if post_type not in words_dict[word].keys():
            words_dict[word][post_type] = 0


"""

    Converts dictionary into a pandas dataframe in order to do operations on the columns which is more efficient
    than doing it on the dictionary.
    :param words_dict

"""


def convert_dict_to_dataframe(words_dict):
    list_for_df = []

    for word in words_dict:
        if "story" in words_dict[word].keys():
            entity_story = words_dict[word]["story"]
        else:
            entity_story = 0
        if "ask_hn" in words_dict[word].keys():
            entity_ask_hn = words_dict[word]["ask_hn"]
        else:
            entity_ask_hn = 0
        if "show_hn" in words_dict[word].keys():
            entity_show_hn = words_dict[word]["show_hn"]
        else:
            entity_show_hn = 0
        if "poll" in words_dict[word].keys():
            entity_poll = words_dict[word]["poll"]
        else:
            entity_poll = 0

        list_for_df.append([word, entity_story, entity_ask_hn, entity_show_hn, entity_poll])

    df = pd.DataFrame(list_for_df, columns=['word', 'story', 'ask_hn', 'show_hn', 'poll'])
    df["story"] = pd.to_numeric(df["story"])
    df.sort_values(by=['word'], ignore_index=True)

    return df


"""

    Gets then sum of the entries in the specified class column.
    (Can refactored in counting in dict instead of creating a dataframe)
    :param df
    :param post_type : class

"""


def get_frequency(df, post_type):
    counter = df[post_type].sum()
    return counter


"""

    Calculates the smoothed conditional probability of each word of the vocabulary based on its class
    :param words_dict
    :param post_type : class
    :param dataframe

"""


def get_conditional_probability(words_dict, dataframe, post_type):
    vocab_size = len(words_dict)
    sum_frequencies_every_word_in_post_type = get_frequency(dataframe, post_type)
    for word in words_dict:
        freq_word_in_post_type = words_dict[word][post_type]
        freq_word_in_post_type_smoothed = freq_word_in_post_type + 0.5
        name_of_key = 'P(w|' + post_type + ')'
        words_dict[word][name_of_key] = freq_word_in_post_type_smoothed / (
                sum_frequencies_every_word_in_post_type + vocab_size * 0.5)

    return words_dict


"""

    Generates the model file.
    Each line of the file contain a word of the vocabulary along with its total frequency and its conditional 
    probabilities for each specific class
    :param file_name
    :param words_dict

"""


def generate_model_file(file_name, words_dict):
    print("Generating model file : ", file_name)
    file = open(file_name, "w")
    counter = 1

    for word in sorted(words_dict):
        wi = word
        frequency_of_wi_in_story = str(words_dict[wi]['story'])
        smoothed_conditional_prop_of_wi_in_story = str(words_dict[wi]['P(w|story)'])
        frequency_of_wi_in_ask_hn = str(words_dict[wi]['ask_hn'])
        smoothed_conditional_prop_of_wi_in_ask_hn = str(words_dict[wi]['P(w|ask_hn)'])
        frequency_of_wi_in_show_hn = str(words_dict[wi]['show_hn'])
        smoothed_conditional_prop_of_wi_in_show_hn = str(words_dict[wi]['P(w|show_hn)'])
        frequency_of_wi_in_poll = str(words_dict[wi]['poll'])
        smoothed_conditional_prop_of_wi_in_poll = str(words_dict[wi]['P(w|poll)'])
        line_to_add = str(
            counter) + "  " + wi + "  " + frequency_of_wi_in_story + "  " + smoothed_conditional_prop_of_wi_in_story + "  " + frequency_of_wi_in_ask_hn + "  " + smoothed_conditional_prop_of_wi_in_ask_hn + "  " + frequency_of_wi_in_show_hn + "  " + smoothed_conditional_prop_of_wi_in_show_hn + "  " + frequency_of_wi_in_poll + "  " + smoothed_conditional_prop_of_wi_in_poll + "\n"

        file.write(line_to_add)
        counter += 1

    file.close()


"""

    Generates the vocabulary file.
    Each line of the file contain a word of the vocabulary.
    :param words_dict

"""


def generate_vocab_file(words_dict):
    print("Generating vocabulary file : vocabulary.txt")
    file = open("vocabulary.txt", "w")
    counter = 1

    for word in sorted(words_dict):
        line_to_add = str(
            counter) + "  " + word + "\n"
        file.write(line_to_add)
        counter += 1

    file.close()


"""

    Generates the list of words/characters excluded from the vocabulary.

"""


def generate_remove_word_file():
    print("Generating remove word file : remove_word.txt")
    file = open("remove_word.txt", "w")
    list = ["\"", "+", "?", "!", ".", ",", ":", ")", "(", "-", "'"]
    counter = 1
    for word in sorted(list):
        line_to_add = str(
            counter) + "  " + word + "\n"
        file.write(line_to_add)
        counter += 1

    file.close()


"""

    Generates the priors from the testing data.
    :param df2018 : dataframe containing training data.

"""


def generate_priors(df2018):
    dict_priors = {}

    size = len(df2018['Post Type'])

    dict_priors['story'] = len(df2018[df2018['Post Type'] == "story"]) / size
    dict_priors['ask_hn'] = len(df2018[df2018['Post Type'] == "ask_hn"]) / size
    dict_priors['show_hn'] = len(df2018[df2018['Post Type'] == "show_hn"]) / size
    dict_priors['poll'] = len(df2018[df2018['Post Type'] == "poll"]) / size

    return dict_priors


"""

    Generates the score of each title for each class using naive bays theorem based on the precomputed model.
    Estimates if the original class given is the same one as the one predicted by the model and specify it using
    right or wrong.
    All the result for each title are stored in the dict_titles_testing dictionary..
    :param titles
    :param dict_titles_testing
    :param dict_priors
    :param words_dict: contains the conditional probabilities of each word in our model

"""


def classify_using_naive_bays(titles, dict_titles_testing, dict_priors, words_dict):
    title_score_dict = nd.nested_dict()
    # counter_right = 0
    # counter_wrong = 0

    for title in titles:
        score_dict = {}
        words = regex_filtering(title.lower())

        if dict_priors['story'] == 0:
            score_story = float('-inf')
        else:
            score_story = math.log10(dict_priors['story'])

        if dict_priors['ask_hn'] == 0:
            score_ask_hn = float('-inf')
        else:
            score_ask_hn = math.log10(dict_priors['ask_hn'])

        if dict_priors['show_hn'] == 0:
            score_show_hn = float('-inf')
        else:
            score_show_hn = math.log10(dict_priors['show_hn'])

        if dict_priors['poll'] == 0:
            score_poll = float('-inf')
        else:
            score_poll = math.log10(dict_priors['poll'])

        for word in words:
            if word in words_dict.keys():  # should we do something if word is not in model or just skip it?
                score_story += math.log10(words_dict[word]['P(w|story)'])
                score_ask_hn += math.log10(words_dict[word]['P(w|ask_hn)'])
                score_show_hn += math.log10(words_dict[word]['P(w|show_hn)'])
                score_poll += math.log10(words_dict[word]['P(w|poll)'])

        score_dict['story'] = score_story
        score_dict['ask_hn'] = score_ask_hn
        score_dict['show_hn'] = score_show_hn
        score_dict['poll'] = score_poll

        max_value = max(score_dict.values())  # maximum value
        max_keys = [k for k, v in score_dict.items() if v == max_value]  # getting all keys containing the `maximum`
        estimated_post_type = max_keys[0]

        title_score_dict[title]['estimation'] = estimated_post_type

        if estimated_post_type == dict_titles_testing[title]:
            title_score_dict[title]['original_post_type'] = dict_titles_testing[title] + ' right'
            # counter_right += 1
        else:
            title_score_dict[title]['original_post_type'] = dict_titles_testing[title] + ' wrong'
            # counter_wrong += 1

        title_score_dict[title]['score_story'] = score_story
        title_score_dict[title]['score_ask_hn'] = score_ask_hn
        title_score_dict[title]['score_show_hn'] = score_show_hn
        title_score_dict[title]['score_poll'] = score_poll

    # print(counter_right)
    # print(counter_wrong)
    return title_score_dict


"""

    Generates the baseline file.
    Each line of the file contain a title along with its score per class and the estimation of the right/wrong class.
    :param filename
    :param title_score_dict
    

"""


def generate_baseline_file(file_name, title_score_dict):
    print("Generating baseline file : ", file_name)
    text_file = open(file_name, "w")
    counter = 1
    for title in title_score_dict:
        original_post_type = str(title_score_dict[title]['original_post_type'])
        score_of_story = str(title_score_dict[title]['score_story'])
        score_of_ask_hn = str(title_score_dict[title]['score_ask_hn'])
        score_of_show_hn = str(title_score_dict[title]['score_show_hn'])
        score_of_poll = str(title_score_dict[title]['score_poll'])
        estimation_post_type = str(title_score_dict[title]['estimation'])

        line_to_add = str(
            counter) + "  " + title + "  " + estimation_post_type + "  " + score_of_story + "  " + score_of_ask_hn + "  " + score_of_show_hn + "  " + score_of_poll + "  " + original_post_type + "\n"

        text_file.write(line_to_add)
        counter += 1

    text_file.close()


"""

    Returns from vocabulary words with a range of specific frequencies.
    :param frequencies_dict
    :param frequency

"""


def get_from_vocabulary_words_with_frequency(frequencies_dict, frequency):
    frequencies_dict_bigger_than = dict(frequencies_dict)
    for word in frequencies_dict:
        if frequencies_dict[word] > frequency:
            frequencies_dict_bigger_than.pop(word, None)

    return frequencies_dict_bigger_than


"""

    Returns from vocabulary top x% in terms of frequency words.
    :param frequencies_dict
    :param percentage

"""


def get_from_vocabulary_most_frequent_words(frequencies_dict, percentage):
    frequencies_dict_most_frequent = dict(frequencies_dict)
    length_dict = len(frequencies_dict)
    number_of_words_to_remove = length_dict * percentage

    i = length_dict
    for word in frequencies_dict:
        if i > number_of_words_to_remove:
            frequencies_dict_most_frequent.pop(word, None)
        i = i - 1

    return frequencies_dict_most_frequent


"""

    Reads csv and returns a dataframe.
    :param filename

"""


def get_dataframe_from_csv(file_name):
    return pd.read_csv(file_name)


"""

    Returns the training data from the dataframe
    :param df

"""


def get_training_data(df):
    return df[df['Created At'].str.contains("2018")]


"""

    Returns the testing data from the dataframe
    :param df

"""


def get_testing_data(df):
    return df[df['Created At'].str.contains("2019")]


"""

    Returns the accuracy, f-mesure, recall and precision of each experiment using the sklearn API.
    :param performance_dict
    :param y_true : the true classes
    :param y_prediction : the estimated classes
    :param nb_of_word_left_in_vocab

"""


def get_performance(performance_dict, y_true, y_prediction, nb_of_word_left_in_vocab):
    # In the multi-class and multi-label case, this is the average of the F1 score of each class with weighting
    # depending on the average parameter.
    # average = 'weighted' : Calculate metrics for each label, and find their average weighted by support

    performance_dict[nb_of_word_left_in_vocab]['accuracy'] = sklearn.metrics.accuracy_score(y_true, y_prediction)
    performance_dict[nb_of_word_left_in_vocab]['f-mesure'] = sklearn.metrics.f1_score(y_true, y_prediction,
                                                                                      average='weighted')
    performance_dict[nb_of_word_left_in_vocab]['recall'] = sklearn.metrics.recall_score(y_true, y_prediction,
                                                                                        average='weighted')
    performance_dict[nb_of_word_left_in_vocab]['precision'] = sklearn.metrics.precision_score(y_true, y_prediction,
                                                                                              average='weighted')

    return performance_dict


"""

    Returns the true classes along with the estimated classes.
    :param title_score_dict

"""


def get_y_true_y_prediction(title_score_dict):
    prediction_list = []
    original_list = []
    result = []

    for title in title_score_dict:
        prediction = title_score_dict[title]["estimation"]
        original = title_score_dict[title]["original_post_type"].split(' ')
        prediction_list.append(prediction)
        original_list.append(original[0])

    result.append(prediction_list)
    result.append(original_list)

    return result


"""

    Gets duplicated titles in dataframe.
    :param df2019

"""


def get_duplicated_titles(df):
    duplicateDFRow = df[df.duplicated(['Title'])]
    print("duplicated", duplicateDFRow.tit)


"""

    Core function that will link all the operations together.
    :param file_name : file from which we read our data from
    :param stop_words : words that we exclude from our vocabulary
    :param model_output : name of model file
    :param baseline_output : name of baseline file

"""


def main(file_name, stop_words, model_output, baseline_output):
    # Read in the survey CSV
    df = get_dataframe_from_csv(file_name)

    # Create training data
    df2018 = get_training_data(df)

    # Create classes list
    classes = ['story', 'ask_hn', 'show_hn', 'poll']

    # Dictionary that will hold all necessary data related to word
    # such as : frequency and conditional probability for each class
    words_dict = nd.nested_dict()

    # Training data for classes

    for class_name in classes:
        df2018_class_name = df2018[df2018['Post Type'] == class_name]

        list_of_story_title = df_title_to_list(df2018_class_name)

        words_dict = title_to_vocab(list_of_story_title, class_name, words_dict)

    # For experiment 1.3.1 and 1.3.3
    # Remove list of predefined words from vocabulary
    for word in stop_words:
        words_dict.pop(word, None)

    # For experiment 1.3.2
    if model_output == "wordlength-model.txt":
        words_dict_iterator = dict(words_dict)  # recreating another copy of the dict for iteration because size changes
        for word in words_dict_iterator:
            if len(word) <= 2 or len(word) >= 9:
                words_dict.pop(word, None)

    # Fill missing values for specific class in dictionary for each word frequency with 0
    # To prevent calculation errors later
    for class_name in classes:
        fill_non_existing_values(words_dict, class_name)

    # Convert dict of frequencies to pandas dataframe to simplify later calculations
    df_word_frequencies = convert_dict_to_dataframe(words_dict)

    # generate smoothed conditinal probabilities for each word in class
    for class_name in classes:
        words_dict = get_conditional_probability(words_dict, df_word_frequencies, class_name)

    # generate model output file
    if len(model_output) > 0:
        generate_model_file(model_output, words_dict)

    # PART 2

    # Create testing data
    df2019 = get_testing_data(df)

    # Find all duplicate rows (titles)
    # get_duplicated_titles(df2019)

    # Getting priors of testing set
    dict_priors = generate_priors(df2018)

    # Transforming column of pandas dataframe such as key: title, value: post Type
    # del dict
    dict_titles_testing = dict(zip(df2019['Title'], df2019['Post Type']))

    # Initializing dictionary that will hold score of each title for each class with prediction
    title_score_dict = nd.nested_dict()

    # Getting list of titles
    list_of_title = df_title_to_list(df2019)

    # Classification using Naive Bays Classification :
    title_score_dict = classify_using_naive_bays(list_of_title, dict_titles_testing, dict_priors, words_dict)

    # Generating Baseline File
    if len(baseline_output) > 0:
        generate_baseline_file(baseline_output, title_score_dict)

    # Return prediction and estimation list to calculate performance
    if len(baseline_output) == 0:
        result = get_y_true_y_prediction(title_score_dict)
        return result

    if baseline_output == "baseline-result.txt":
        result = get_y_true_y_prediction(title_score_dict)
        result.append(len(words_dict))
        generate_vocab_file(words_dict)
        generate_remove_word_file()
        return result


# Specify filename
file_name = "hns_2018_2019.csv"

# initialize dictionary of performance when removing top x%,
# Key: words left in vocab Value: precision, recall, accuracy, f-mesure
performance_in_percent_dict = nd.nested_dict()

# initialize dictionary of performance when removing words with x frequency ,
# Key: words left in vocab Value: precision, recall, accuracy, f-mesure
performance_dict = nd.nested_dict()

# Part 1 & Part 2
stop_words = []
baseline_performance = main(file_name, stop_words, "model-2018.txt", "baseline-result.txt")
performance_dict = get_performance(performance_dict, baseline_performance[1], baseline_performance[0],
                                   baseline_performance[2])
performance_in_percent_dict = get_performance(performance_in_percent_dict, baseline_performance[1],
                                              baseline_performance[0], baseline_performance[2])

# 1.3.1 Experiment 1: Stop-word Filtering

# Reading stopwords file and storing each word in a list by specifying the return to line delimiter
text_file = open("stopwords.txt", "r")
stop_words = text_file.read().split('\n')
main(file_name, stop_words, "stopword-model.txt", "stopword-result.txt")

# 1.3.2 Experiment 2: Word Length Filtering
stop_words = []
main(file_name, stop_words, "wordlength-model.txt", "wordlength-result.txt")

# 1.3.3 Experiment 3: Infrequent Word Filtering
frequencies_dict = {}

df = get_dataframe_from_csv(file_name)

df2018 = get_training_data(df)

list_of_title = df_title_to_list(df2018)

# Populating frequencies dictionary
for title in list_of_title:
    words = regex_filtering(title.lower())
    for word in words:
        if word in frequencies_dict.keys():
            frequencies_dict[word] = frequencies_dict[word] + 1
        else:
            frequencies_dict[word] = 1

# sort dictionary according to frequencies of words
frequencies_dict = {k: v for k, v in sorted(frequencies_dict.items(), key=lambda item: item[1])}
size_dict = len(frequencies_dict)

# get from the vocabulary words with frequency=1
frequencies_dict_equal_to_1 = get_from_vocabulary_words_with_frequency(frequencies_dict, 1)
stop_words_1 = list(frequencies_dict_equal_to_1.keys())
number_of_words_left_in_vocab = size_dict - len(frequencies_dict_equal_to_1)
result_1 = main(file_name, stop_words_1, "", "")
performance_dict = get_performance(performance_dict, result_1[1], result_1[0], number_of_words_left_in_vocab)

# get from the vocabulary words with frequency<=5
frequencies_dict_smaller_equal_5 = get_from_vocabulary_words_with_frequency(frequencies_dict, 5)
stop_words_5 = list(frequencies_dict_smaller_equal_5.keys())
number_of_words_left_in_vocab = size_dict - len(frequencies_dict_smaller_equal_5)
result_5 = main(file_name, stop_words_5, "", "")
performance_dict = get_performance(performance_dict, result_5[1], result_5[0], number_of_words_left_in_vocab)

# get from the vocabulary words with frequency<=10
frequencies_dict_smaller_equal_10 = get_from_vocabulary_words_with_frequency(frequencies_dict, 10)
stop_words_10 = list(frequencies_dict_smaller_equal_10.keys())
number_of_words_left_in_vocab = size_dict - len(frequencies_dict_smaller_equal_10)
result_10 = main(file_name, stop_words_10, "", "")
performance_dict = get_performance(performance_dict, result_10[1], result_10[0], number_of_words_left_in_vocab)

# get from the vocabulary words with frequency<=15
frequencies_dict_smaller_equal_15 = get_from_vocabulary_words_with_frequency(frequencies_dict, 15)
stop_words_15 = list(frequencies_dict_smaller_equal_15.keys())
number_of_words_left_in_vocab = size_dict - len(frequencies_dict_smaller_equal_15)
result_15 = main(file_name, stop_words_15, "", "")
performance_dict = get_performance(performance_dict, result_15[1], result_15[0], number_of_words_left_in_vocab)

# get from the vocabulary words with frequency<=20
frequencies_dict_smaller_equal_20 = get_from_vocabulary_words_with_frequency(frequencies_dict, 20)
stop_words_20 = list(frequencies_dict_smaller_equal_20.keys())
number_of_words_left_in_vocab = size_dict - len(frequencies_dict_smaller_equal_20)
result_20 = main(file_name, stop_words_20, "", "")
performance_dict = get_performance(performance_dict, result_20[1], result_20[0], number_of_words_left_in_vocab)

# remove from the vocabulary 5 % most frequent words
frequencies_dict_top_5 = get_from_vocabulary_most_frequent_words(frequencies_dict, 0.05)
stop_words_top_5 = list(frequencies_dict_top_5.keys())
number_of_words_left_in_vocab = size_dict - len(frequencies_dict_top_5)
result_top_5 = main(file_name, stop_words_top_5, "", "")
performance_in_percent_dict = get_performance(performance_in_percent_dict, result_top_5[1], result_top_5[0],
                                              number_of_words_left_in_vocab)

# remove from the vocabulary 10 % most frequent words
frequencies_dict_top_10 = get_from_vocabulary_most_frequent_words(frequencies_dict, 0.1)
stop_words_top_10 = list(frequencies_dict_top_10.keys())
number_of_words_left_in_vocab = size_dict - len(frequencies_dict_top_10)
result_top_10 = main(file_name, stop_words_top_10, "", "")
performance_in_percent_dict = get_performance(performance_in_percent_dict, result_top_10[1], result_top_10[0],
                                              number_of_words_left_in_vocab)

# remove from the vocabulary 15 % most frequent words
frequencies_dict_top_15 = get_from_vocabulary_most_frequent_words(frequencies_dict, 0.15)
stop_words_top_15 = list(frequencies_dict_top_15.keys())
number_of_words_left_in_vocab = size_dict - len(frequencies_dict_top_15)
result_top_15 = main(file_name, stop_words_top_15, "", "")
performance_in_percent_dict = get_performance(performance_in_percent_dict, result_top_15[1], result_top_15[0],
                                              number_of_words_left_in_vocab)

# remove from the vocabulary 20 % most frequent words
frequencies_dict_top_20 = get_from_vocabulary_most_frequent_words(frequencies_dict, 0.2)
stop_words_top_20 = list(frequencies_dict_top_20.keys())
number_of_words_left_in_vocab = size_dict - len(frequencies_dict_top_20)
result_top_20 = main(file_name, stop_words_top_20, "", "")
performance_in_percent_dict = get_performance(performance_in_percent_dict, result_top_20[1], result_top_20[0],
                                              number_of_words_left_in_vocab)

# remove from the vocabulary 25 % most frequent words
frequencies_dict_top_25 = get_from_vocabulary_most_frequent_words(frequencies_dict, 0.25)
stop_words_top_25 = list(frequencies_dict_top_25.keys())
number_of_words_left_in_vocab = size_dict - len(frequencies_dict_top_25)
result_top_25 = main(file_name, stop_words_top_25, "", "")
performance_in_percent_dict = get_performance(performance_in_percent_dict, result_top_25[1], result_top_25[0],
                                              number_of_words_left_in_vocab)

# Preparing values for first graph
x_values = list(performance_dict.keys())

accuracy = []
precision = []
fmesure = []
recall = []

for i in x_values:
    accuracy.append(performance_dict[i]["accuracy"])
    precision.append(performance_dict[i]["precision"])
    recall.append(performance_dict[i]["recall"])
    fmesure.append(performance_dict[i]["f-mesure"])

# Preparing values for second graph
x_values_per = list(performance_in_percent_dict.keys())

accuracy_per = []
precision_per = []
fmesure_per = []
recall_per = []

for i in x_values_per:
    accuracy_per.append(performance_in_percent_dict[i]["accuracy"])
    precision_per.append(performance_in_percent_dict[i]["precision"])
    recall_per.append(performance_in_percent_dict[i]["recall"])
    fmesure_per.append(performance_in_percent_dict[i]["f-mesure"])

# Defining figure size
fig = plt.figure(figsize=(10, 7))

# Plotting the first graph
ax = fig.add_subplot(221)

ax.plot(x_values, precision, marker='*', color='k')
ax.plot(x_values, fmesure, marker='v', color='b')
ax.plot(x_values, recall, marker='+', color='y', linestyle='dashed', zorder=1)
ax.plot(x_values, accuracy, marker='o', color='r', zorder=0)

# Legend of first graph
plt.title('Performance for frequency of words')
ax.set_ylabel("Percentage")
ax.set_xlabel("Number of words left in vocabulary")

# Plotting the second graph
ax2 = fig.add_subplot(222)

ax2.plot(x_values_per, precision_per, marker='*', color='k')
ax2.plot(x_values_per, fmesure_per, marker='v', color='b')
ax2.plot(x_values_per, recall_per, marker='+', color='y', linestyle='dashed', zorder=1)
ax2.plot(x_values_per, accuracy_per, marker='o', color='r', zorder=0)

# Legend of second graph
plt.title('Performance for percentage of words')
ax2.set_ylabel("Percentage")
ax2.set_xlabel("Number of words left in vocabulary")

# Common legend
v = mlines.Line2D([], [], color='blue', marker='v', linestyle='None',
                  markersize=10, label='F-mesure')
o = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                  markersize=10, label='Accuracy')
plus = mlines.Line2D([], [], color='y', marker='+', linestyle='None',
                     markersize=10, label='Recall')

star = mlines.Line2D([], [], color='k', marker='*', linestyle='None',
                     markersize=10, label='Precision')
ax.legend(handles=[v, o, plus, star], bbox_to_anchor=(1.3, -0.2))

print('Thanks for executing the code, see you next time!')
plt.show()