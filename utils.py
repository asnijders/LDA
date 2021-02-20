import pandas as pd

class DataModule():
    """
    This class loads (a subset) of the corpus
    """
    def __init__(self, num_docs):

        """
        This function initializes the datamodule class
        :param num_docs: number of documents to load from json
        """
        self.num_docs = num_docs
        self.data = pd.read_json('data/arxiv-metadata-oai-snapshot.json', lines=True,
                                   nrows=self.num_docs)
        # create a list of list of documents from the abstracts, lowercase words
        self.docs = [ [word.lower() for word in doc.split() if word.isalpha()] for doc in self.data['abstract'].tolist()]
        self.vocab = self.get_vocab() # obtain vocabulary
        self.docs2ids() # convert words to ids

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

    def docs2ids(self):
        """
        This function replaces each word in the corpus with the integer in the vocabulary associated with that word
        :return:
        """
        self.docs = [ [self.vocab[word] for word in doc] for doc in self.docs]