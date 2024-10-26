import re
import numpy as np

from transformers import AutoTokenizer, AutoModelForTokenClassification


class NER():
    """
    Named Entity Recognition (NER) class for identifying named entities in Kazakh text.
    This class utilizes a pre-trained NER model to recognize various entity types such as persons, locations, organizations, etc.
    """
    def __init__(self):
        """
        Initializes the NER model and tokenizer.
        Loads the pre-trained NER model and tokenizer from the specified checkpoint.
        """

        self.model_checkpoint = "weights/checkpoint-8.0"

        self.labels_dict = {0:"O", 1:"B-ADAGE", 2:"I-ADAGE", 3:"B-ART", 4:"I-ART", 5:"B-CARDINAL",
                6:"I-CARDINAL", 7:"B-CONTACT", 8:"I-CONTACT", 9:"B-DATE", 10:"I-DATE", 11:"B-DISEASE",
                12:"I-DISEASE", 13:"B-EVENT", 14:"I-EVENT", 15:"B-FACILITY", 16:"I-FACILITY",
                17:"B-GPE", 18:"I-GPE", 19:"B-LANGUAGE", 20:"I-LANGUAGE", 21:"B-LAW", 22:"I-LAW",
                23:"B-LOCATION", 24:"I-LOCATION", 25:"B-MISCELLANEOUS", 26:"I-MISCELLANEOUS",
                27:"B-MONEY", 28:"I-MONEY", 29:"B-NON_HUMAN", 30:"I-NON_HUMAN", 31:"B-NORP",
                32:"I-NORP", 33:"B-ORDINAL", 34:"I-ORDINAL", 35:"B-ORGANISATION", 36:"I-ORGANISATION",
                37:"B-PERCENTAGE", 38:"I-PERCENTAGE", 39:"B-PERSON", 40:"I-PERSON", 41:"B-POSITION",
                42:"I-POSITION", 43:"B-PRODUCT", 44:"I-PRODUCT", 45:"B-PROJECT", 46:"I-PROJECT",
                47:"B-QUANTITY", 48:"I-QUANTITY", 49:"B-TIME", 50:"I-TIME"}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_checkpoint)

    def predict(self, input_sent):
        """
        Predicts named entities in the given input sentence.

        Args:
            input_sent (str, optional): The input sentence to be processed. Defaults to a sample Kazakh sentence.

        Returns:
            dict: A dictionary mapping tokens to their corresponding entity labels.
        """
        
        tokenized_inputs = self.tokenizer(input_sent, return_tensors="pt")

        output = self.model(**tokenized_inputs)
        predictions = np.argmax(output.logits.detach().numpy(), axis=2)

        word_ids = tokenized_inputs.word_ids(batch_index=0)
        previous_word_idx = None
        labels = []
        for i, p in zip(word_ids, predictions[0]):
            # Special tokens have a word id that is None. We set the label to -100 so they are
            # automatically ignored in the loss function.
            if i is None or i == previous_word_idx:
                continue
            elif i != previous_word_idx:
                try:
                    labels.append(self.labels_dict[p][2:])
                except:
                    labels.append(self.labels_dict[p])
            previous_word_idx = i

        input_sent_tokens = re.findall(r"[\w’-]+|[.,#?!)(\]\[;:–—\"«№»/%&']", input_sent)
        assert len(input_sent_tokens) == len(labels), "Mismatch between input token and label sizes!"
        result_dict = {}
        for t,l in zip(input_sent_tokens, labels):
            result_dict[t] = l
        return result_dict
