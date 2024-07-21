from transformer_models import ByT5

def binary_search(arr, low, high, x):
 
    # Check base case
    if high >= low:
 
        mid = (high + low) // 2
        # If element is present at the middle itself
        if arr[mid] == x:
            
            return mid
 
        # If element is smaller than mid, then it can only
        # be present in left subarray
        elif arr[mid] > x:
            return binary_search(arr, low, mid - 1, x)
 
        # Else the element can only be present in right subarray
        else:
            return binary_search(arr, mid + 1, high, x)
 
    else:
        # Element is not present in the array
        return -1


class DetectLanguageBaseline:
    def __init__(self, words_path="mixed_languages/words.txt", model = None):
        
        with open(words_path, "r", encoding="utf-8") as file:
            self.words = file.readlines()
            self.words = [word.strip() for word in self.words]
            self.words.sort()

            self.model = model

    def detect_english(self, word):
        """
        Detects the language of a word based on the words list

        Args:
            word (str): The word to detect the language of

        Returns:
            int: If the word is found in the list, returns the index of the word in the list, 
                otherwise returns -1
        """
        
        ret = binary_search(self.words, 0, len(self.words), word)

        return ret
    
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

            segment_data = {}
            segment_data['lang'] = 'el'
            segment_data['text'] = array[start_index:index]

            segmented_array.append(segment_data)
            
            segment_data = {}
            segment_data['lang'] = 'en'
            segment_data['text'] = [array[index]]

            segmented_array.append(segment_data)

            start_index = index + 1

        if(start_index < len(array)):
            segment_data = {}
            segment_data['lang'] = 'el'
            
            segment_data['text'] = array[start_index:]

            segmented_array.append(segment_data)

        
        
        return segmented_array
    
    def __call__(self, text):
        
        english_words_indices = []
        words = text.split(" ")

        # Detect the english words
        for i, word in enumerate(words):
            indx = self.detect_english(word)
            if(indx != -1):
                english_words_indices.append(i)

        # Split the array based on the positions of the english words
        text_split = self.split_array(words, english_words_indices)


        predicted_text = []

        for segment in text_split:
            if(len(segment['text']) == 0):
                continue
            txt = " ".join(segment['text'])
            
            if segment['lang'] == 'en':
                predicted_text.extend([txt])
            else:
                predicted_text.extend([self.model(txt)])

        predicted_text = " ".join(predicted_text)
        
        return predicted_text
                




        


    
        

if __name__ == "__main__":

    model = ByT5.ByT5Model("AUEB-NLP/ByT5_g2g")
    model.eval()

    baseline = DetectLanguageBaseline(words_path="mixed_languages/words.txt", model=model)


    # print(baseline.detect_english("what"))
    text = baseline("prepei na kanoume adjust sta kainouria guidelines epeidh to evaluation tha rthei kai tha einai poly arnhtiko")
    
    
    print(text)
    
