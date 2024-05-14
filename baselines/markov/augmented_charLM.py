"""
This model combines the ngram character model with a word-LSTM language model. The scores are calculated as a weighted
sum of the scores of the charLM and the wordLM at each step of the translation of the sentence.
"""
from torch import nn as nn
import torch
from baselines.dictionary_tables import greeklish_to_greek_intonated
from baselines.RNN.util import *
from baselines.RNN.LSTM_LM import LSTM_LangModel
from evaluate import load
from math import log

class State:
    """
        Class that contains all the info of a state of the translation as it moves. A typical state consists of:
            * A string of the remaining sentence in greeklish that needs to be translated
            * A list of greek character tokens: The sentence translated so far, used by the charLM
            * A list of greek word tokens: The sentence so far, used by the wordLM(may not be needed)
            * The hidden states of the LSTM, so that we don't need to start the wordLSTM from the beginning every time
            * The combined score of the two models, given as: Score = λ1*charLM + λ2*wordLM, where
             λ1 & λ2 are configurable
            (may need to keep the word score, as it changes les)
    """

    def __init__(self, translated_chars, translated_words, remaining, lstm_out, lstm_hidden, score):
        self.translated_chars = translated_chars
        self.translated_words = translated_words
        self.remaining_sentence = remaining
        self.lstm_out = lstm_out
        self.lstm_hidden = lstm_hidden
        self.score = score

    def __eq__(self, other):
        """
        Equality operator implementation, needed for eliminating equal candidate states at the end of a step.
        """
        if isinstance(other, State):
            if (self.translated_chars == other.translated_chars and  # no need to check for words too
                    self.remaining_sentence == other.remaining_sentence and
                    # also testing the LSTM output & states is redundant, and very wasteful
                    self.score == other.score):
                return True

        return False


class AugmentedCharLM:

    def __init__(self, n_gram_size, vectorizer, model, l1=0.5):
        self.n_gram_size = n_gram_size
        self.n_gram_counts = dict()
        self.vectorizer = vectorizer
        self.model = model
        # We are forced to use regular softmax here, as the log has to be applied
        # after the scores of the two models are calculated and combined
        self.softmax = nn.Softmax(dim=1)
        assert(1 >= l1 >= 0)
        self.l1 = l1
        self.l2 = 1-l1

    def load_model(self, path):
        """
        Load a pre-trained model as a state dictionary.
        """
        self.model.load_state_dict(torch.load(path))

    def fit(self, corpus):
        """
        Fit the model on a new corpus by calculating new char n-gram counts.
        :param corpus: A body of text in the for of an array of Strings representing sentences in Greek.
        """
        self.n_gram_counts = self.calculate_ngrams(corpus)

    def calculate_ngrams(self, corpus):
        """
        Counts the total appearances for every n_gram in the corpus and returns the (ngram: #appearances) pairs.
        """
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

    def translate(self, sentences, beams):
        """
        :param sentences: list of sentences to be translated
        :param beams: number of beams during translation
        :return: List containing the translated sentences
        """
        self.model.eval()
        with torch.no_grad():
            translated_sentences = []
            for sentence in sentences:
                # Prepare the original state
                translated_chars = self.pad("<s>")
                translated_words = []
                remaining_sentence = sentence
                score = 0
                # First, we need to "prep" the word model. This is done by feeding the network with the
                # <s> token and saving the output hidden states for the initial State() object.
                start_input = self.vectorizer.input_tensor("<s>")
                out, h_n, c_n = self.model(start_input, None, None)

                # Create the initial state, and add to the list of states
                initial_state = State(
                    translated_chars=translated_chars,
                    translated_words=translated_words,
                    remaining=remaining_sentence,
                    lstm_out=out,
                    lstm_hidden=(h_n, c_n),
                    score=0
                )
                states = [initial_state]

                # Main translation loop
                for i in range(len(sentence)):
                    candidates = []
                    # Go through every state in the list
                    for state in states:
                        # For every state generate the candidates and add them to the candidates list
                        candidates.append(self.get_candidates(state))

                    # Remove any duplicate candidates
                    # print([candidate.translated for candidate in candidates])
                    candidates_set = []
                    [candidates_set.append(cand) for cand in candidates if cand not in candidates_set]
                    candidates = candidates_set

                    # Get the best states, according to the number of beams in the search.
                    best_candidates = []
                    for j in range(beams):
                        if candidates:
                            # Probabilities of each candidate new state.
                            probabilities = [cand.score for cand in candidates]
                            # Get the best candidate and remove from the list
                            best_cand = candidates.pop(probabilities.index(max(probabilities)))
                            # Add that candidate to the list of best candidates
                            best_candidates.append(best_cand)
                    # Make the list of the best candidates the new states.
                    states = best_candidates

                    # Once the sentence is over, the state with the highest probability is the best translation.
                    probs = [state.score for state in states]
                    # Extract the sentence from the state
                    sent = states[probs.index(max(probs))]
                    # Convert the list of translated tokens to a sentence.
                    translation = ""
                    for i in sent.translated:
                        translation += i

                    translated_sentences.append(translation)
                return translated_sentences

    def get_candidates(self, state):
        # If the state is already a final state (no remaining text to translate),
        # it returns only itself as a candidate.
        if not state.remaining_sentence:
            return [state]

        # If it is not a final state, generate the next candidate states
        candidates = []

        translated_chars = state.translated_chars
        translated_words = state.translated_words
        remaining_sentence = state.remaining_sentence
        out = state.lstm_out
        (h_n, c_n) = state.lstm_hidden

        # Look at the next 1 & next 2 tokens in the remaining sentence
        for length in [1, 2]:
            if len(remaining_sentence) >= length:
                if length == 2:
                    token = remaining_sentence[0] + remaining_sentence[1]
                    replacements = greeklish_to_greek_intonated.get(token, [])
                else:
                    token = remaining_sentence[0]
                    replacements = greeklish_to_greek_intonated.get(token, [token])
                    # If the token is a space token, it's means a new word has been generated
                    if replacements == [" "]:
                        # Start by splitting the sentence through the vectorizer, omitting the <s> padding at the start
                        words = self.vectorizer.split_sequence("".join(translated_chars[self.n_gram_size:]))
                        # Add the last word to the translated words
                        translated_words.append(words[-1])
                        # Finally, feed the latest word to the model
                        input = self.vectorizer.input_tensor(words[-1])
                        out, h_n, c_n = self.model(input, h_n, c_n)

                # For each candidate replacement, get the n-gram probability
                for item in replacements:
                    if len(item) == 2:
                        # In the case we deal with double greek character tokens e.g. 'e' --> 'αι', or 'ai'--> 'αι'
                        history = translated_chars[-(self.n_gram_size - 2):]
                        ngram = tuple(history + [item[0], item[1]])
                        # Create the new state and add it to the candidates
                        if not translated_words:
                            # If the translated words list is still empty, don't use the word LM to calculate the score
                            score = log(self.n_gram_counts.get(ngram, 0))
                        else:
                            probs = self.softmax(out)
                            idx = self.vectorizer.vocab.get(translated_words[-1], len(self.vectorizer.vocab))
                            score_chars = self.n_gram_counts.get(ngram, 0)
                            score_words = probs[0][idx].item()
                            score = log(self.l1 * score_chars + self.l2 * score_words)
                        new_state = State(translated_chars=state.translated_chars + [item[0], item[1]],
                                          translated_words=translated_words,
                                          remaining=state.remaining_sentence[len(token):],
                                          lstm_out=out,
                                          lstm_hidden=(h_n, c_n),
                                          score=state.score + score)
                    else:
                        history = state.translated_chars[-(self.n_gram_size - 1):]
                        ngram = tuple(history + [item])
                        # Create the new state and add it to the candidates
                        if not translated_words:
                            # If the translated words list is still empty, don't use the word LM to calculate the score
                            score = log(self.n_gram_counts.get(ngram, 0))
                        else:
                            probs = self.softmax(out)
                            idx = self.vectorizer.vocab.get(translated_words[-1], len(self.vectorizer.vocab))
                            score_chars = self.n_gram_counts.get(ngram, 0)
                            score_words = probs[0][idx].item()
                            score = log(self.l1 * score_chars + self.l2 * score_words)
                        new_state = State(translated_chars=state.translated_chars + [item],
                                          translated_words=translated_words,
                                          remaining=state.remaining_sentence[len(token):],
                                          lstm_out=out,
                                          lstm_hidden=(h_n, c_n),
                                          score=state.score + score)
                    candidates.append(new_state)

        return candidates
