import pickle
import difflib
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

from tqdm import tqdm


class CodeSwitching:
    """
    This class implements a model for code switching detection
    """
    def __init__(self):
        """
        Class variable initialization
        """
        self.train_data = None
        self.words = None
        self.n_words = None
        self.n_tags = None
        self.tags = None
        self.model = None

    def train(self, data_path: str):
        """
        Method to train the code switching model
        :param data_path:
            Path to the data file (TSV format)
        """
        col_names = ['tweet_id', 'user_id', 'start', 'end', 'token', 'gold_label']

        self.train_data = pd.read_csv(data_path, sep='\t', names=col_names, header=None)
        self.words = self.train_data['token'].unique().tolist()
        self.words = [str(word) for word in self.words]
        self.words.append("ENDPAD")
        self.n_words = len(self.words)
        print("Length of vocabulary = ", self.n_words)

        self.tags = self.train_data["gold_label"].unique().tolist()
        self.n_tags = len(self.tags)
        print("number of tags = ", self.n_tags)

        tweet_ids = self.train_data['tweet_id'].unique()
        X = []
        Y = []

        for tweet_id in tweet_ids:
            tokens = []
            labels = []
            subset = self.train_data[self.train_data['tweet_id'] == tweet_id].sort_values(ascending=True, by='start')
            for index, row in subset.iterrows():
                tokens.append(row.token)
                labels.append(self.tags.index(row.gold_label))
            X.append(tokens)
            Y.append(labels)

        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(None, 1)))
        self.model.add(TimeDistributed(Dense(1, activation='sigmoid')))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

        for idx in tqdm(range(len(X))):
            x = [str(word) for word in X[idx]]
            num_words = len(x)
            embeddings = [self.words.index(word) for word in x]
            input = np.array(embeddings)
            label = np.array(Y[idx])

            input = input.reshape(1, num_words, 1)
            label = label.reshape(1, num_words, 1)
            self.model.fit(input, label, epochs=1, batch_size=1, verbose=0)

    def predict(self, input: str) -> List:
        """
        Performs code switching analysis of an input string
        :param input:
            Sentence to be analyzed
        :return:
            Two lists containing the detected tokens and the detected language
        """
        splited = input.split()
        input = []
        for word in splited:
            closest = difflib.get_close_matches(word, self.words, n=1)
            input.append(self.words.index(closest[0]))
        input = np.array(input)
        input = input.reshape(1, len(splited), 1)
        result = self.model.predict_classes(input)[0]
        lang = []
        token = []
        for i, val in enumerate(result):
            lang.append(self.tags[val[0]])
            token.append(splited[i])
        return token, lang

    def save_model(self, path):
        """
        Saves the model in pickle format
        :param path:
            Path where to save the model
        """
        with open(path, 'wb') as f:
            pickle.dump([self.model, self.words, self.tags], f)

    def load_model(self, path):
        """
        Loads a model from a pickle file
        :param path:
            Path to the picke file containing the model
        """
        with open(path, 'rb') as f:
            self.model, self.words, self.tags = pickle.load(f)
