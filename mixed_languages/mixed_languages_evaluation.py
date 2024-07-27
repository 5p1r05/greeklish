from evaluate import load
import json
import tqdm 

from transformer_models import ByT5
from mixed_languages.detect_language import DetectLanguageBaseline
from mixed_languages.detect_language import split_array


class MixedLanguagesEvaluator:
    """
    This class it used to evaluate a model's ability to ignore english words when transliterating greeklish text.

    Attributes:
        cer (Metric): The Character Error Rate metric
        wer (Metric): The Word Error Rate metric
        bleu (Metric): The BLEU metric
        greek_mixed_text (list): The greek text with mixed english words
        greeklish_mixed_text (list): The greeklish text with mixed english words
        masked_indices (dict): The indices of the english words in the greeklish text

    """
    
    def __init__(self, greek_mixed_text_path, greeklish_mixed_text_path):
        """
        Initializes the MixedLanguagesEvaluator with the paths to the greek and greeklish mixed text files.

        Args:
            greek_mixed_text_path (str): The path to the greek mixed text file
            greeklish_mixed_text_path (str): The path to the greeklish mixed text file

        """
        self.cer = load("cer")
        self.wer = load("wer")
        self.bleu = load("bleu")

        # Load the files
        with open(greek_mixed_text_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            self.greek_mixed_text = data["text"]
            self.masked_indices = data["masked_indices"]
        
        with open(greeklish_mixed_text_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            self.greeklish_mixed_text = data["text"]
    
    


    def evaluate(self, model):

        """
        Evaluates the model's ability to ignore english words when transliterating greeklish text.
        It first calculated the metrics

        Args:
            model (ByT5): The model to evaluate
        """

        predicted_text = []

        # Predict raw text
        for sentence in self.greeklish_mixed_text:
            predicted_text.append(model(sentence))

        # Calculate raw metrics
        cer_raw = self.cer.compute(predictions=predicted_text, references=self.greek_mixed_text)
        wer_raw = self.wer.compute(predictions=predicted_text, references=self.greek_mixed_text)
        print("Metrics for the raw output")
        print(f"CER_raw: {cer_raw}\n"
              f"WER_raw: {wer_raw}")
        
        

        # Calculate the metrics for the greek parts
        predicted = []

        for sent in predicted_text:
            predicted.append(sent.split(" "))

        
        # Split the sentences into words
        predicted_text_sliced = [sent.split(" ") for sent in predicted_text]
        greeklish_mixed_text_sliced = [sent.split(" ") for sent in self.greeklish_mixed_text]
        
        gt_english_text = []
        predicted_english_text = []

        # Iterate over the masked indices
        for sent_idx in self.masked_indices:

            gt_english_words = []
            predicted_english_words = []

            for word_idx in self.masked_indices[sent_idx]:
                if word_idx >= len(greeklish_mixed_text_sliced[int(sent_idx)]) or word_idx >= len(predicted_text_sliced[int(sent_idx)]):
                    continue
                
                # Get the predictions for the english words and the ground truth english words
                gt_english_word = greeklish_mixed_text_sliced[int(sent_idx)][word_idx]
                predicted_english_word = predicted_text_sliced[int(sent_idx)][word_idx]

                gt_english_words.append(gt_english_word)
                predicted_english_words.append(predicted_english_word)

                # Substitute the ground truth english words with the predicted ones
                predicted_text_sliced[int(sent_idx)][word_idx] = gt_english_word

            gt_english_text.append(" ".join(gt_english_words))
            predicted_english_text.append(" ".join(predicted_english_words))
            
                

        predicted_text = [" ".join(sent) for sent in predicted_text_sliced]

        

        cer_raw = self.cer.compute(predictions=predicted_text, references=self.greek_mixed_text)
        wer_raw = self.wer.compute(predictions=predicted_text, references=self.greek_mixed_text)
        print("Metrics for the greeklish output")
        print(f"CER_raw: {cer_raw}\n"
              f"WER_raw: {wer_raw}")
        

        cer_raw = self.cer.compute(predictions=predicted_english_text, references=gt_english_text)
        wer_raw = self.wer.compute(predictions=predicted_english_text, references=gt_english_text)
        print("Metrics for the english output")
        print(f"CER_raw: {cer_raw}\n"
              f"WER_raw: {wer_raw}")
        


if __name__ == "__main__":

    greek_mixed_text_path = "mixed_languages/data/greek_mixed_europarl_words_4.json"
    greeklish_mixed_text_path = "mixed_languages/data/greeklish_mixed_europarl_words_4.json"

    evaluator = MixedLanguagesEvaluator(greek_mixed_text_path, greeklish_mixed_text_path)

    # Load the model
    model = ByT5.ByT5Model("AUEB-NLP/ByT5_g2g")
    model.eval()

    evaluator.evaluate(model)

    print("Evaluating the baseline")

    baseline = DetectLanguageBaseline(words_path="mixed_languages/words.txt", model=model)

    evaluator.evaluate(baseline)




