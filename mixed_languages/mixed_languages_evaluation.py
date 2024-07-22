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

        # Calculate metrics
        cer_raw = self.cer.compute(predictions=predicted_text, references=self.greek_mixed_text)
        wer_raw = self.wer.compute(predictions=predicted_text, references=self.greek_mixed_text)
        print("raw output")
        print(f"CER_raw: {cer_raw}\n"
              f"WER_raw: {wer_raw}")
        # print("raw output")
        # print(predicted_text)
        
        
        greeklish_split = self.greeklish_mixed_text.copy()

        # Split the sentences into words
        for i in range(len(greeklish_split)):
            greeklish_split[i] = greeklish_split[i].split(" ")
      
        # Split the subsentences that contain english words
        for idx in range(len(greeklish_split)):
            if(str(idx) in self.masked_indices.keys()):
                greeklish_split[idx] = split_array(greeklish_split[idx], self.masked_indices[str(idx)])
            else:
                greeklish_split[idx] = [{'lang': 'el', 'text':greeklish_split[idx]}]

        predicted_text = []

        for sentence in greeklish_split:

            predicted_sentence  = []
            for segment in sentence:
                if(len(segment['text']) == 0):
                    continue
                txt = " ".join(segment['text'])
                
                if segment['lang'] == 'en':
                    predicted_sentence.extend([txt])
                else:
                    predicted_sentence.extend([model(txt)])
            
            predicted_text.append(" ".join(predicted_sentence))
           
           
        
        cer_sub = self.cer.compute(predictions=predicted_text, references=self.greek_mixed_text)
        wer_sub = self.wer.compute(predictions=predicted_text, references=self.greek_mixed_text)
        print("substituted output")
        print(f"CER_sub: {cer_sub}\n"
              f"WER_sub: {wer_sub}")
        
        print(f"CER difference: {cer_raw - cer_sub}")
        print(f"WER difference: {wer_raw - wer_sub}")



if __name__ == "__main__":

    greek_mixed_text_path = "mixed_languages/data/greek_mixed_europarl_words_4.json"
    greeklish_mixed_text_path = "mixed_languages/data/greeklish_mixed_europarl_words_4.json"

    evaluator = MixedLanguagesEvaluator(greek_mixed_text_path, greeklish_mixed_text_path)

    # Load the model
    model = ByT5.ByT5Model("AUEB-NLP/ByT5_g2g")
    model.eval()

    evaluator.evaluate(model)

    baseline = DetectLanguageBaseline(words_path="mixed_languages/words.txt", model=model)

    evaluator.evaluate(baseline)




