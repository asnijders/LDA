from utils import DataModule
import numpy as np

class LDA():

    def __init__(self, dataset, num_topics, alpha, beta):
        """
        This class implements Latent Dirichlet Allocation using only Python routines
        :param dataset: datamodule object
        :param num_topics: number of clusters
        :param alpha: parameter for dirichlet
        :param beta: parameter for dirichlet
        """

        # data specific attributes
        self.docs = dataset.docs # list of documents
        self.num_docs = dataset.num_docs
        self.vocab = dataset.vocab # dictionary with words2ids
        self.id2word = {v: k for k, v in self.vocab.items()} # dictionary with ids2words
        self.vocab_size = len(self.vocab.keys())

        # model specific parameters
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta

        # for each word in each document, randomly assign a topic out of the possible topic choices
        self.topic_assignments = [[np.random.randint(0, self.num_topics) for word in doc] for doc in self.docs]

        # initialise counter variables with zero
        self.n_doc_topic = np.zeros((self.num_docs,
                                     self.num_topics))
        self.n_topic_word = np.zeros((self.num_topics,
                                      self.vocab_size))
        self.n_topic = np.zeros((self.num_topics))

        # Now increment the counters with previously randomly initialised topic assignments
        for d, doc in enumerate(self.topic_assignments):
            for assignment in doc:
                self.n_doc_topic[d][assignment] += 1
                self.n_topic[assignment] += 1

        # Also increment counters to reflect how many times a word i was assigned to topic k
        for d, doc in enumerate(self.docs):
            for i, word in enumerate(doc):
                topic = self.topic_assignments[d][i]
                self.n_topic_word[topic][word] += 1

        # initialise topic distribution
        self.topics = np.zeros((self.num_topics))

    def train(self, iterations, eval_every, num_words):
        """
        This function implements posterior inference to train LDA using Gibbs sampling
        :param iterations: number of iterations over corpus
        :param eval_every: interval at which topics are printed
        :param num_words: amount of words printed per evaluation
        :return: None
        """

        for m in range(iterations): # over m iterations

            print('Beginning iteration: {}'.format(m))

            for d in range(self.num_docs): # over d documents in corpus

                for i in range(len(self.docs[d])): # over i words per document d

                    # we look at the i-th word of the d-th document; we also pick the corresponding topic
                    word = self.docs[d][i]
                    topic = self.topic_assignments[d][i]

                    # we then decrement the counter variables for this document-topic, topic-word combination
                    # to temporarily move them 'out' of the equation
                    self.n_doc_topic[d][topic] -= 1
                    self.n_topic_word[topic][word] -= 1
                    self.n_topic[topic] -= 1


                    for k in range(0, self.num_topics):

                        lhs = (self.n_doc_topic[d][k] + self.alpha)
                        numerator = (self.n_topic_word[k][word] + self.beta)
                        denominator = (self.n_topic[k] + (self.beta * self.vocab_size) )
                        rhs = numerator / denominator
                        self.topics[k] =  lhs * rhs

                    # Here we apply a normalization step over the topic array such that we can sample
                    self.topics = [value/sum(self.topics) for value in self.topics]
                    topic = np.random.multinomial(1, pvals=self.topics).argmax()
                    self.topic_assignments[d][i] = topic

                    # increment the counter variables
                    self.n_doc_topic[d][topic] += 1
                    self.n_topic_word[topic][word] += 1
                    self.n_topic[topic] += 1

            if m % eval_every == 0: # print topics every <eval_every> iterations
                self.top_words(num_words=num_words)

    def top_words(self, num_words):
        """
        This function prints the top <num_words> per topic
        :param num_words:
        :return: None
        """

        for k in range(1):
            normalized_topic = [count/sum(self.n_topic_word[k]) for count in self.n_topic_word[k]]
            top_word_ids = np.asarray(normalized_topic).argsort()[-num_words:][::-1]

            print('words for 1st topic: ')
            print([self.id2word[id] for id in top_word_ids])

if __name__ == '__main__':

    # load first 500 documents from corpus
    dataset = DataModule(num_docs=500)
    # initialise LDA model
    LDAmodel = LDA(dataset=dataset,
                   num_topics=25,
                   alpha=1,
                   beta=1)
    # Perform posterior inference
    LDAmodel.train(iterations=500,
                   eval_every=25,
                   num_words =20)
    