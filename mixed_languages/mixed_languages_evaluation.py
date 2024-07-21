from evaluate import load
import json
import tqdm 

from transformer_models import ByT5


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
    
    def split_array(self, array, indices):
        """
        Splits an array into segments based on the indices

        Args:
            array (list): The array to split
            indices (list): The indices to split the array
        
        Returns:
            list: The segmented array
        """
        segmented_array = []

        start_index = 0

        for index in indices:
            segmented_array.append(array[start_index:index])
            segmented_array.append([array[index]])

            start_index = index + 1

        if(start_index < len(array)):
            segmented_array.append(array[start_index:])
        
        
        return segmented_array


    def evaluate(self, model):

        """
        Evaluates the model's ability to ignore english words when transliterating greeklish text.
        It first calculated the metrics 
        """

        predicted_text = []

        # Predict raw text
        for sentence in self.greeklish_mixed_text:
            predicted_text.append(model(sentence))

        # Calculate metrics
        cer = self.cer.compute(predictions=predicted_text, references=self.greek_mixed_text)
        wer = self.wer.compute(predictions=predicted_text, references=self.greek_mixed_text)
        print(f"CER_raw: {cer}"
              f"WER_raw: {wer}")
        
        greeklish_split = self.greeklish_mixed_text.copy()

        # Split the sentences into words
        for i in range(len(greeklish_split)):
            greeklish_split[i] = greeklish_split[i].split(" ")
      
        # Split the subsentences that contain english words
        for idx in range(len(greeklish_split)):
            if(str(idx) in self.masked_indices.keys()):
                greeklish_split[idx] = self.split_array(greeklish_split[idx], self.masked_indices[str(idx)])
            else:
                greeklish_split[idx] = [greeklish_split[idx]]

        transliterated = []

        # Transliterate the subsentences
        for sentence_split in greeklish_split:

            sentence_join = []
            for segment in sentence_split:
                if(len(segment) == 0):
                    continue
                output = model(" ".join(segment))
                sentence_join.extend([output.split(" ")])

            sentence_join = [item for sublist in sentence_join for item in sublist]
            transliterated.append(sentence_join)

        
        for sent_idx in self.masked_indices:
          for idx in self.masked_indices[sent_idx]:
            transliterated[int(sent_idx)][idx] = self.greek_mixed_text[int(sent_idx)].split(" ")[idx]

        transliterated = [" ".join(sent) for sent in transliterated]
        print(transliterated)
        
        cer = self.cer.compute(predictions=transliterated, references=self.greek_mixed_text)
        wer = self.wer.compute(predictions=transliterated, references=self.greek_mixed_text)
        print(f"CER_raw: {cer}"
              f"WER_raw: {wer}")


       


if __name__ == "__main__":

    greek_mixed_text_path = "mixed_languages/data/greek_mixed_europarl_words_20.json"
    greeklish_mixed_text_path = "mixed_languages/data/greeklish_mixed_europarl_words_20.json"

    evaluator = MixedLanguagesEvaluator(greek_mixed_text_path, greeklish_mixed_text_path)

    # Load the model
    model = ByT5.ByT5Model("AUEB-NLP/ByT5_g2g")
    model.eval()

    evaluator.evaluate(model)

