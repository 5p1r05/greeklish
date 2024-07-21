import os
import json
import random
import googletrans

import dataset_maker


def mix_texts(greek_text_path, english_text_path, num_sentences, substitution_probability):
    """
    Mixes two aligned texts, greek and english, by randomly substituting sentences from the english text to the greek text.

    Args:
        greek_text_path (str): Path to the greek text file.
        english_text_path (str): Path to the english text file.
        num_sentences (int): Number of sentences.
        substitution_probability (float): Probability of substituting an english sentence to the greek text.

    Returns:
        dict: A dictionary containing the mixed text and the indices of the english sentences that were substituted.
    """

    greek_english_data = {}

    # Read the files
    with open(greek_text_path, "r") as f:
        greek_text = f.readlines()

    with open(english_text_path, "r") as f: 
        english_text = f.readlines()

    greek_english_data['text'] = []
    greek_english_data['masked_indices'] = []

    # Iterate over the sentences and randomly substitute 
    # greek sentences with the corresponding english ones
    for i in range(num_sentences):
        if(random.random() < substitution_probability):
            greek_english_data['text'].append(english_text[i])
            greek_english_data['masked_indices'].append(i)
        else:
            greek_english_data['text'].append(greek_text[i])

    return greek_english_data

def substitute_words(greek_text_path, num_sentences, substitution_probability):
    """
    Substitutes words in a greek text with random english words.

    Args:
        greek_text_path (str): Path to the greek text file.
        substitution_probability (float): Probability of substituting a word.

    Returns
    """
    with open(greek_text_path, "r") as f:
        greek_text = f.readlines()

    # Load the translator
    translator = googletrans.Translator()
    greek_english_data = {}
    greek_english_data['text'] = []
    greek_english_data['masked_indices'] = {}
    

    for i in range(num_sentences):
        word_counter = 0
        words = greek_text[i].split()
        for j, word in enumerate(words):
            if(random.random() < substitution_probability):
                translation = translator.translate(word, src='el', dest='en').text
                words[j] = translation

                print(f"{i} : {translation}")

                if(i not in greek_english_data['masked_indices'].keys()):
                    greek_english_data['masked_indices'][i] = []


                greek_english_data['masked_indices'][i].extend(list(range(word_counter, word_counter + len(translation.split(" ")))))
                word_counter += len(translation.split(" "))
            else:
                word_counter += 1

        greek_english_data['text'].append(" ".join(words))
        

    return greek_english_data


    


# def convert_to_greeklish_selected(greek_english_data):
#     """
#     Changes the greek sentences in the mixed text to greeklish.

#     Args:
#         greek_english_data (dict): A dictionary containing the mixed text and the indices of the english sentences that were substituted.

#     Returns:
#         dict: A dictionary containing the mixed greeklish-english text.
#     """
#     sentences = greek_english_data['text']
#     masked_indices = greek_english_data['masked_indices']

#     greeklish_english_data = {}
#     greeklish_english_data['text'] = []
#     greeklish_english_data['masked_indices'] = masked_indices

#     for i, sentence in enumerate(sentences):
#         if i not in masked_indices:
#             greeklish_english_data['text'].append(dataset_maker.convert_to_greeklish([sentence])[0])
#         else:
#             greeklish_english_data['text'].append(sentence)
    
#     return greeklish_english_data




if __name__ == "__main__":
    greek_text_path = "mixed_languages/data/europarl-v7.el-en.el"
    english_text_path = "mixed_languages/data/europarl-v7.el-en.en"

    num_sentences = 20
    substitution_probability = 0.15
    random.seed(123)

    greek_english_sentences = mix_texts(greek_text_path, english_text_path, num_sentences, substitution_probability)

    with open(f"mixed_languages/data/greek_mixed_europarl_sentences_{num_sentences}.json", "w", encoding ='utf8') as f:
        json.dump(greek_english_sentences, f, ensure_ascii=False)


    # Create the greeklish version of the mixed text
    # greeklish_english_sentences = convert_to_greeklish_selected(greek_english_sentences)
    greek_english_sentences['text'] = dataset_maker.convert_to_greeklish(greek_english_sentences['text'])

    with open(f"mixed_languages/data/greeklish_mixed_europarl_sentences_{num_sentences}.json", "w", encoding ='utf8') as f:
        json.dump(greek_english_sentences, f, ensure_ascii=False)

    # Substitute words
    greek_english_words = substitute_words(greek_text_path, num_sentences, substitution_probability)

    with open(f"mixed_languages/data/greek_mixed_europarl_words_{num_sentences}.json", "w", encoding ='utf8') as f:
        json.dump(greek_english_words, f, ensure_ascii=False)

    # Create the greeklish version of the mixed text
    greek_english_words['text'] = dataset_maker.convert_to_greeklish(greek_english_words['text'])

    with open(f"mixed_languages/data/greeklish_mixed_europarl_words_{num_sentences}.json", "w", encoding ='utf8') as f:
        json.dump(greek_english_words, f, ensure_ascii=False)