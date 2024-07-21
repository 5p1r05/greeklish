# from evaluate import load
import random
from dictionary_tables import greek_to_greeklish_intonated


def convert_to_greeklish(texts_greek, indices=None):
    """
    This function takes a list of original Greek sentences, and converts them to Greeklish, using the conversions
    stored in the dictionaries above. Each time a token in a sentence has more than 1 possible conversions (for example,
    η -> {h, i} ), the sentence branches into two (or more, depending on the amount of possible conversions) new
    sentences, equivalent up to that point, and the translation continues independently for each branch. This ensures
    that each possible way to convert a sentence to Greeklish gets an equal representation in the resulting dataset.
    --------------------
    @:param texts_greek: a string array containing the corpus in greek, with each row representing a distinct sentence
    @:return texts_greeklish: a string array containing the converted sentences from greek_texts
    """

    # We start by checking for tokens in this order:
    #       1)double_tokens
    #       2)special_tokens
    #       3)simple_tokens

    texts_greeklish = []
    random.seed(12300)
    for sentence in texts_greek:
        converted_sent = ""
        i = 0

        while i < len(sentence):
            # If the current token read belongs to double_token conversions
            if(i+1 <= len(sentence)-1 )and (sentence[i] + sentence[i+1] in greek_to_greeklish_intonated.keys()):
                # If there are more than 1 possible conversions, pick one of them randomly
                if isinstance(greek_to_greeklish_intonated.get(sentence[i] + sentence[i + 1]), list):
                    index = random.randrange(len(greek_to_greeklish_intonated.get(sentence[i] + sentence[i + 1])))
                    converted_sent += greek_to_greeklish_intonated.get(sentence[i] + sentence[i + 1])[index]
                else:
                    converted_sent += greek_to_greeklish_intonated.get(sentence[i] + sentence[i + 1])
                i += 2
            # If the current token read belongs to special_token conversions
            elif sentence[i] in greek_to_greeklish_intonated.keys():
                if isinstance(greek_to_greeklish_intonated.get(sentence[i]), list):
                    index = random.randrange(len(greek_to_greeklish_intonated.get(sentence[i])))
                    converted_sent += greek_to_greeklish_intonated.get(sentence[i])[index]
                else:
                    converted_sent += greek_to_greeklish_intonated.get(sentence[i])
                i += 1
            # If the current token does not belong to the Greek alphabet, transfer it directly
            else:
                converted_sent += sentence[i]
                i += 1

        # Add the converted sentence to the list
        texts_greeklish.append(converted_sent)
    return texts_greeklish



if __name__ == "__main__":
    with open(data_path+"greek_europarl_test_5k.txt", "r", encoding="utf-8") as file:
        tests = []
        for line in file:
            tests.append(line[:-1])

    processed = convert_to_greeklish(tests)

    with open(data_path+"greeklish_europarl_test_5k.txt", "w", encoding="utf-8") as file:
        for i in processed:
            file.write(i)
            file.write("\n")

