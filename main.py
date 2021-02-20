from utils import DataModule

import numpy as np
import argparse

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

            if m % eval_every == 0 and m != 0: # print topics every <eval_every> iterations
                self.top_words(num_words=num_words)

    def top_words(self, num_words):
        """
        This function prints the top <num_words> per topic
        :param num_words:
        :return: None
        """

        for k in range(3):
            normalized_topic = [count/sum(self.n_topic_word[k]) for count in self.n_topic_word[k]]
            top_word_ids = np.asarray(normalized_topic).argsort()[-num_words:][::-1]

            print('words for 1st topic: ')
            print([self.id2word[id] for id in top_word_ids])

def main(config):

    # print CLI args
    print(' ')
    for arg in vars(args):
        print(str(arg) + ': ' + str(getattr(args, arg)))
    print(' ')

    # load first 500 documents from corpus
    dataset = DataModule(num_docs=config.num_docs,
                         threshold=config.threshold)

    print('Vocabulary size: {} unique tokens\n'.format(len(dataset.vocab)))

    # initialise LDA model
    LDAmodel = LDA(dataset=dataset,
                   num_topics=config.num_topics,
                   alpha=config.alpha,
                   beta=config.beta)
    # Perform posterior inference
    LDAmodel.train(iterations=config.iterations,
                   eval_every=config.eval_every,
                   num_words =config.num_words)

if __name__ == '__main__':

    # Feel free to add more argument parameters
    config = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data related parameters
    config.add_argument('--num_docs', default=500, type=int,
                        help='number of documents to read')
    config.add_argument('--threshold', default=100, type=int,
                        help='words that occur more than x times get removed')

    # model related parameters
    config.add_argument('--num_topics', default=25, type=int,
                        help='Number of topics to be learned')
    config.add_argument('--alpha', default=1.0, type=float,
                        help='Alpha parameter')
    config.add_argument('--beta', default=0.02, type=float,
                        help='Beta parameter')

    # train related parameters
    config.add_argument('--iterations', default=500, type=int,
                        help='Number of iterations over corpus')
    config.add_argument('--eval_every', default=5, type=int,
                        help='Print topics at intervals of ..')
    config.add_argument('--num_words', default=20, type=int,
                        help='Number of words to print per topic')

    args = config.parse_args()

    main(args)