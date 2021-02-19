import pandas as pd

class datamodule():

    def __init__(self, num_docs):

        self.num_docs = num_docs
        self.data = pd.read_json('data/arxiv-metadata-oai-snapshot.json', lines=True,
                                   nrows=self.num_docs)

        self.docs = [ [word.lower() for word in doc.split() if word.isalpha()] for doc in self.data['abstract'].tolist()]
        self.vocab = self.get_vocab()
        self.docs2ids()

    def get_vocab(self):
        word2id = {}
        for document in self.docs:
            for word in document:
                if word not in word2id.keys():
                    word2id[word] = len(word2id)
        return word2id

    def docs2ids(self):
        self.docs = [ [self.vocab[word] for word in doc] for doc in self.docs]