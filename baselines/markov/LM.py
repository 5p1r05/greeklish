import pickle
from baselines.dictionary_tables import greeklish_to_greek, greeklish_to_greek_intonated
import random
from evaluate import load
import unicodedata as ud

def load_model(filename):
    """
    Loads a pretrained LM saved as a pickle file, e.g. >>> lm = models.load("2_gram_char_lm.pkl")
    :param filename: The path to the file
    :return:
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


class LM:

    def __init__(self, beam_parameters=2, n_gram_size=2, intonation=False ):
        self.name = str(n_gram_size)+"_gram_char_lm"
        self.n_gram_size = n_gram_size
        self.n_gram_counts = {}
        self.beam_parameters = beam_parameters
        if intonation:
            self.dictionary = greeklish_to_greek_intonated
        else:
            self.dictionary = greeklish_to_greek

    def fit(self, corpus):
        """
        Fit the model on a new corpus by calculating new char n-gram counts.
        :param corpus: A body of text in the for of an array of Strings representing sentences in Greek.
        """
        self.n_gram_counts = self.__calculate_ngrams(corpus)

    def predict(self, sentences):
        """
        Takes a corpus of test sentences in Greeklish and translates them tho Greek,
        based on the statistical data obtained from fitting on the training corpus.
        :param sentences: String array of sentences in Greeklish
        :return: String of sentences translated into Greek
        """
        translations = []
        for sentence in sentences:
            translations.append(self.translate_sentence(sentence))

        return translations

    def translate_sentence(self, sentence):
        """
        Returns the Greek translation with the highest probability for the given
        sentence in Greeklish.
        :param sentence: String representing original sentence in Greeklish
        :returns String of the Greek translation of @:sentence
        """
        # Add <start> padding at the beginning of the sentence.
        sentence = self.pad('<s>') + [char for char in sentence]
        # The states that will be stored to perform the beam search.
        # Each state will be a tuple: (<generated sentence>,<remaining sentence>,<probability>)
        # The initial state's generated sentence is just the '<s>' padding before the sentence.
        # The initial state's probability must be neutral, so it gets initialized as 1.
        states = [(sentence[:self.n_gram_size-1], sentence[self.n_gram_size-1:], 1)]

        for i in range(self.n_gram_size-1, len(sentence)):
            candidates = []
            # Look through the current states.
            for state in states:
                # Produce the next-char-candidates from each state, along with their probabilities.
                new_states = self.get_candidates(state)
                candidates += new_states

            # Remove any duplicate candidates
            candidates_set = []
            [candidates_set.append(cand) for cand in candidates if cand not in candidates_set]
            candidates = candidates_set
            # Get the best states, according to the number of parameters in the beam search.
            best_candidates = []
            for j in range(self.beam_parameters):
                if candidates:
                    # Probabilities of each candidate new state.
                    probabilities = [cand[2] for cand in candidates]
                    # Get the best candidate and remove from the list
                    best_cand = candidates.pop(probabilities.index(max(probabilities)))
                    # Add that candidate to the list of best candidates
                    best_candidates.append(best_cand)
            # Make the list of best candidates the new states
            states = best_candidates

        # Once the sentence is over, the state with the highest probability is the best translation.
        probs = [state[2] for state in states]
        # Extract the sentence from the state
        sent = states[probs.index(max(probs))]
        sent = sent[0][self.n_gram_size-1:]
        translation = ""
        for i in sent:
            translation += i

        return translation

    def get_candidates(self, state):
        # If the state is already a final state, it returns only
        # itself as a candidate
        if not state[1]:
            return [state]

        # If it is not a final state, generate the next candidate states
        translated_sentence = state[0]
        remaining_sentence = state[1]
        probability = state[2]
        candidates = []

        # Look at both of the first two characters in the translated sentence, as some greek
        # characters may be represented with 2 characters in Greeklish.
        for length in [1, 2]:
            if len(remaining_sentence) >= length:
                # Fetch the valid replacements from the dictionary.
                if length == 2:
                    token = remaining_sentence[0]+remaining_sentence[1]
                    replacements = self.dictionary.get(token, [])
                else:
                    # If the look-up is a miss (e.g. from a space or a number),
                    # return the character itself.
                    token = remaining_sentence[0]
                    replacements = self.dictionary.get(token, [token])

                # For each candidate replacement, get the probability from the LM
                for item in replacements:
                    if len(item) == 2:
                        # In the case we deal with double greek character tokens e.g. 'e' --> 'αι', or 'ai'--> 'αι'
                        history = translated_sentence[-(self.n_gram_size - 2):]
                        ngram = tuple(history + [item[0], item[1]])
                        candidates.append((translated_sentence + [item[0], item[1]],
                                           remaining_sentence[len(token):],
                                           self.n_gram_counts.get(ngram, 0)))
                    else:
                        history = translated_sentence[-(self.n_gram_size - 1):]
                        ngram = tuple(history + [item])
                        candidates.append((translated_sentence + [item],
                                           remaining_sentence[len(token):],
                                           self.n_gram_counts.get(ngram, 0)))
        return candidates

    def __calculate_ngrams(self, corpus):
        counts = {}
        count_total = 0
        for sentence in corpus:
            # Tokenize and add <start> & <end> padding to the sentence
            sent = self.pad('<s>') + [token for token in sentence]

            # Create the n-grams of the sentence
            for i in range(len(sent) - (self.n_gram_size - 1)):
                n_gram = tuple(sent[i: i + self.n_gram_size])
                # Update the count for each n-gram observed
                if n_gram not in counts.keys():
                    counts[n_gram] = 1

                else:
                    counts[n_gram] += 1
                count_total += 1

        for key in counts.keys():
            counts[key] /= count_total
        return counts

    def pad(self, symbol):
        """
        Used to apply the padding at the beginning and ending of each n-gram.
        """
        return (self.n_gram_size - 1) * [symbol]

    def save(self):
        with open(self.name+".pkl", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

# Testing pipeline for the model
def main():
    path = "/europarl\\"

    with open(path+"greeklish_europarl_v7.txt", encoding="utf-8") as file:
        texts_greeklish = ["" + line.strip() + "" for line in file]

    with open(path+"greek_europarl_v7.txt", encoding="utf-8") as file:
        texts_greek = ["" + line.strip() + "" for line in file]

    # Shuffle both corpuses on the same seed
    random.seed(12300)
    random.shuffle(texts_greek)

    random.seed(12300)
    random.shuffle(texts_greeklish)

    # Testing set (5000 sentences)
    test_x = texts_greeklish[500000:505000]
    # Labels with intonation
    test_y_intonated = texts_greek[500000:505000]
    # Non-intonated labels
    test_y_not_intonated = []
    d = {ord('\N{COMBINING ACUTE ACCENT}'): None}
    for s in texts_greek[500000:505000]:
        test_y_not_intonated.append(ud.normalize('NFD', s).translate(d))


    # Character Error Rate metric
    cer = load("cer")
    wer = load("wer")
    bleu = load("bleu")
    results = []
    with open("../results_LM_new.txt", 'w', encoding="utf-8") as file:

        for intonation in [True]:
            if intonation:
                file.write("Results with Intonation:")
                file.write("\n")
            else:
                file.write("Results without Intonation:")
                file.write("\n")

            for n_gram_size in [7]:
                for beam_params in [6,7,8]:
                    # Configure model
                    model = LM(beam_parameters=beam_params, n_gram_size=n_gram_size, intonation=intonation)

                    for training_size in [30000]:
                        train = texts_greek[:training_size]
                        model.fit(train)
                        predictions = model.predict(test_x)
                        if intonation:
                            references = test_y_intonated
                        else:
                            references = test_y_not_intonated
                        cer_score = cer.compute(predictions=predictions, references=references)
                        wer_score = wer.compute(predictions=predictions, references=references)
                        bleu_score = bleu.compute(predictions=predictions, references=[[x] for x in references])
                        results.append(cer_score)
                        file.write("{}-grams_{}_beams_{}: {} {} {}".format(n_gram_size,beam_params,training_size, round(cer_score, 4), round(wer_score,4), round(bleu_score["bleu"],4)))
                        file.write("\n")
    # Save result list
    with open("../results_LM_new.pkl", "wb") as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()


"""
with open("train_10k.pkl", "rb") as f:
    train = pickle.load(f)
sentences = ["Ο ουρανός είναι μπλε και ο κορονοιός επεκτείνεται"]
greeklish = ["O uranos einai mple kai o koronoios epekteinetai"]

model = LM(intonation=True, n_gram_size=6, beam_parameters=5)
model.fit(train)
print(model.predict(greeklish))
print(sentences)
"""