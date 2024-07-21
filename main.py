from datetime import datetime
import pickle
from dictionary_tables import greek_to_greeklish_intonated
from models.RNN.LSTM_LM import LSTM_LangModel
from models.RNN.main import LanguageModel
from models.RNN.util import *
import torch
from evaluate import load

models_path = "models/"
data_path = "data/"
results_path = "results/"

cer = load("cer")
wer = load("wer")
bleu = load("bleu")

# Load the test data
with open(data_path + "greeklish_europarl_test_5k.txt", "r", encoding="utf-8") as file:
    test_sources = []
    for line in file:
        # REMOVE newline
        line = line[:-1]
        # REVERSE ORDER
        test_sources.append(line[::-1])

with open(data_path + "greek_europarl_test_5k.txt", "r", encoding="utf-8") as file:
    test_targets = []
    for line in file:
        # REMOVE newline
        line = line[:-1]
        # REVERSE ORDER
        test_targets.append(line[::-1])


# Reverse models
model_params = [(32,512,2),(32,512,5),(64,512,2),(64,512,5),(64,1024,5),(128,300,5),(128,512,5)]
counter = 0
for params in model_params:

    # Load model
    input_size = 120
    embed_size = params[0]
    hidden_size = params[1]
    output_size = 120

    model = LSTM_LangModel(input_size, embed_size, hidden_size, output_size)
    path = "models/LSTM_models_trained/rev_1layer_LSTM_LM_50000_char_120_{}_{}/0{}_dropout/".format(
        embed_size, hidden_size, params[2])

    model.load_state_dict(
        torch.load(path+"rev_1layer_LSTM_LM_50000_char_120_{}_{}.pt".format(embed_size, hidden_size), map_location=torch.device('cpu')))

    with open(path+"rev_vectorizer_50000_char_120_{}_{}.pickle".format(embed_size, hidden_size), "rb") as f:
        text_vec = pickle.load(f)

    LM = LanguageModel(text_vec, model)

    print("Successful load. Translation began. {}".format(datetime.now()))

    # Number of beams does matter.
    results = LM.translate(test_sources[:1000], 9)

    scores = []

    scores.append(cer.compute(predictions=results, references=test_targets[:1000]))
    scores.append(wer.compute(predictions=results, references=test_targets[:1000]))
    scores.append(bleu.compute(predictions=results, references=test_targets[:1000]).get("bleu"))

    # write the resulting translations
    with open(results_path+"results_rev_1layer_50000_char_120_{}_{}_0{}drop.txt".format(embed_size, hidden_size, params[2]), "w", encoding="utf-8") as file:

            file.write("Test Score at [:1000] of greeklish_europarl_test_5k.txt")
            file.write('\n')
            file.write("CER: {}".format(scores[0]))
            file.write('\n')
            file.write("WER: {}".format(scores[1]))
            file.write('\n')
            file.write("BLEU: {}".format(scores[2]))
            file.write('\n')

    counter += 1
    print(str(counter)+"/7 done. {}".format(datetime.now()))