import pandas as pd
from nltk.stem.porter import PorterStemmer

class DataModule():
    """
    This class loads (a subset) of the corpus
    """
    def __init__(self, num_docs, upper, lower, stemmer=True):
        """
        This function initializes the datamodule class
        :param num_docs: number of documents to load from json
        """

        # load data from json file
        self.num_docs = num_docs
        self.data = pd.read_json('data/arxiv-metadata-oai-snapshot.json', lines=True,
                                   nrows=self.num_docs)

        # create a list of list of documents from the abstracts, lowercase words, remove non-alphanumerical words
        self.stemmer = PorterStemmer() if stemmer == True else None
        self.docs = [self.preprocess(doc) for doc in self.data['abstract'].tolist()]

        # count occurrences per word and prune words that occur above some threshold
        self.counts = self.get_counts()
        self.prune(upper=upper, lower=lower)

        # obtain 'final' vocabulary, convert to integer ids.
        self.vocab = self.get_vocab()
        self.docs2ids()

    def get_vocab(self):
        """
        this function creates a vocabulary given the corpus, assigns integers to words
        :return: dictionary vocabulary object
        """
        word2id = {}
        for document in self.docs:
            for word in document:
                if word not in word2id.keys():
                    word2id[word] = len(word2id)
        return word2id

    def get_counts(self):
        """
        this function creates a vocabulary given the corpus, assigns integers to words
        :return: dictionary vocabulary object
        """
        counts = {}
        for document in self.docs:
            for word in document:
                if word not in counts.keys():
                    counts[word] = 1
                else:
                    counts[word] += 1
        return counts

    def prune(self, upper, lower):
        """
        this function removes all words that occur more than some threshold
        :return:
        """
        # max_count = sorted([self.counts[key] for key in self.counts.keys()])[::-1][upper]
        max_count = upper

        print('Removed all words that occur less than {} times and more than {} times'.format(lower, upper))
        for i, doc in enumerate(self.docs):
            new_doc = []
            for word in doc:
                if self.counts[word] <= max_count and self.counts[word] > lower:
                    new_doc.append(word)
            self.docs[i] = new_doc

    def docs2ids(self):
        """
        This function replaces each word in the corpus with the integer in the vocabulary associated with that word
        :return:
        """
        self.docs = [ [self.vocab[word] for word in doc] for doc in self.docs]

    def preprocess(self, doc):
        """
        this function lower-cases words and stems them
        :param doc:
        :return:
        """
        self.stopwords = []
        doc = doc.split()
        new_doc = []
        for word in doc:
            if word.isalpha() and word.lower() not in self.stopwords:
                word = word.lower()
                if self.stemmer != None:
                    word = self.stemmer.stem(word)
                new_doc.append(word)
        return new_doc
