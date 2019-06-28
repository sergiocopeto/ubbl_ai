from typing import List

import pandas as pd
import re
import nltk

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle


class LanguageIdentifier:
    """
    This class implements a model for language identification, being able to be trained with any language or language variant
    """
    def __init__(self):
        """
        Variable initialization. Also checks if nltk library has all the files needed to process the data
        """
        nltk.download('punkt')
        self.data = None
        self.codes = []
        self.tfidf_vect = None
        self.multinomial_nb = None
        self.model = None

    def load_data(self, path: List[str], code: List[str], num_samples_per_file: int = -1) -> pd.DataFrame:
        """
        Data loading process
        :param path:
            List of paths to the different language data files
        :param code:
            List of language codes to be classified
        :param num_samples_per_file:
            Number of samples to be collected per input file (-1 means that the method will read all the lines)
        :return:
            DataFrame containing all read data
        """
        data_dict = {
            'code': [],
            'code_num': [],
            'sentence': []
        }
        for idx, language_file in enumerate(path):
            if code[idx] not in self.codes:
                self.codes.append(code[idx])
            code_num = self.codes.index(code[idx])
            language_code = self.codes[code_num]
            with open(language_file, encoding='utf8') as f:
                if num_samples_per_file > 0:
                    lines = [f.readline() for x in range(num_samples_per_file)]
                else:
                    lines = f.readlines()
                for line in lines:
                    data_dict['code'].append(language_code)
                    data_dict['code_num'].append(code_num)
                    data_dict['sentence'].append(line)

        if self.data is None:
            self.data = pd.DataFrame.from_dict(data_dict)
        else:
            self.data.append(pd.DataFrame.from_dict(data_dict))

        return self.data

    def tokenize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform sentence tokenization
        :param data:
            DataFrame containing the data to be processed
        :return:
            DataFrame with an added column with the tokenized sentence
        """
        to_process = data.copy()
        to_process['tokens'] = to_process['sentence'].apply(sent_tokenize)
        return to_process

    def _clean(self, sentences: List, pattern = r'<(!?).*>') -> str:
        """
        Performs string cleanup
        :param sentences:
            Sentences to be clean
        :param pattern:
            Pattern for sentence cleaning
        :return:
            Cleaned sentence
        """
        for i, sentence in enumerate(sentences):
            sentences[i] = re.sub(pattern, '', sentence)
        return sentences[0]

    def clean_sentences(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Performs sentence cleaning on source DataFrame
        :param data:
            DataFrame containing all the data to be processed
        :return:
            DataFrame containing an additional column with the cleaned sentence
        """
        to_process = data.copy()
        to_process['clean'] = to_process['tokens'].apply(self._clean)
        return to_process

    def fit(self, train_data: List, train_labels: List, n_gram_range: tuple = (1, 3)):
        """
        Trains the language detection model
        :param train_data:
            List of sentences as training data
        :param train_labels:
            List of corresponding gold labels
        :param n_gram_range:
            length range of the n-grams to be computed
        """
        train_data, train_labels = shuffle(train_data, train_labels, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(train_data,
                                                            train_labels,
                                                            test_size=0.3,
                                                            random_state=42)
        self.tfidf_vect = TfidfVectorizer(analyzer='char', ngram_range=n_gram_range)
        self.multinomial_nb = MultinomialNB()
        self.model = Pipeline([('tfidf', self.tfidf_vect),
                             ('multinomial_nb', self.multinomial_nb),
                             ])
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test,predictions)
        print('Trained model with accuracy value of:', accuracy)
        print(classification_report(y_test, predictions, target_names=self.codes))

    def train(self, path: List[str], code: List[str], num_samples_per_file=100, n_gram_range=(1, 3)):
        """
        Full training pipeline, combining all the above methods
        :param path:
            List of paths to the training data
        :param code:
            List of language codes to be detected
        :param num_samples_per_file:
            Number of samples to be collected per input file (-1 means that the method will read all the lines)
        :param n_gram_range:
            length range of the n-grams to be computed
        :return:
        """
        self.load_data(path, code, num_samples_per_file)
        self.data = self.tokenize(self.data)
        self.data = self.clean_sentences(self.data)
        self.fit(self.data.clean, self.data.code_num, n_gram_range)

    def predict(self, input: str, use_model_codes: bool = False) -> List:
        """
        Performs language detection of the input string
        :param input:
            Input Sentence to be classified
        :param use_model_codes:
            Boolean that returns a list of the actual language codes provided by the model if set to true
        :return:
        """
        if isinstance(input, str):
            result = self.model.predict([input])
        elif isinstance(input, List):
            result = self.model.predict(input)
        output = []
        if use_model_codes:
            for res in result:
                output.append(self.codes[res])
            return output
        return result