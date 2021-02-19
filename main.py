from utils import datamodule

import numpy as np

class LDA():

    def __init__(self, dataset, num_topics, alpha, beta):

        # data specific attributes
        self.docs = dataset.docs
        self.num_docs = dataset.num_docs
        self.vocab = dataset.vocab
        self.id2word = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab.keys())

        # model specific parameters
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta

        # randomly initialise topic assignments per word
        self.topic_assignments = [[np.random.randint(0, self.num_topics) for word in doc] for doc in self.docs]

        # initialise counters
        self.n_doc_topic = np.zeros((self.num_docs,
                                     self.num_topics))
        self.n_topic_word = np.zeros((self.num_topics,
                                      self.vocab_size))
        self.n_topic = np.zeros((self.num_topics))

        # increment counters with previously randomly initialised topic assignments
        for d, doc in enumerate(self.topic_assignments):
            for assignment in doc:
                self.n_doc_topic[d][assignment] += 1
                self.n_topic[assignment] += 1

        # increment counters to reflect how many times a word i was assigned to topic k
        for d, doc in enumerate(self.docs):
            for i, word in enumerate(doc):
                topic = self.topic_assignments[d][i]
                self.n_topic_word[topic][word] += 1

        # initialise topic distribution
        self.topics = np.zeros((self.num_topics))

    def train(self, iterations, eval_every, num_words):

        for m in range(iterations): # over m iterations

            print('Beginning iteration: {}'.format(m))

            for d in range(self.num_docs): # over d documents in corpus

                for i in range(len(self.docs[d])): # over i words per document d

                    word = self.docs[d][i]
                    topic = self.topic_assignments[d][i]

                    self.n_doc_topic[d][topic] -= 1
                    self.n_topic_word[topic][word] -= 1
                    self.n_topic[topic] -= 1
                    
                    for k in range(0, self.num_topics):

                        lhs = (self.n_doc_topic[d][k] + self.alpha)
                        numerator = (self.n_topic_word[k][word] + self.beta)
                        denominator = (self.n_topic[k] + (self.beta * self.vocab_size) )
                        rhs = numerator / denominator
                        self.topics[k] =  lhs * rhs

                    # apply a normalization step?
                    self.topics = [value/sum(self.topics) for value in self.topics]

                    topic = np.random.multinomial(1, pvals=self.topics).argmax()
                    self.topic_assignments[d][i] = topic

                    self.n_doc_topic[d][topic] += 1
                    self.n_topic_word[topic][word] += 1
                    self.n_topic[topic] += 1

            if m % eval_every == 0:
                self.top_words(num_words=num_words)


    def top_words(self, num_words):

        for k in range(1):
            normalized_topic = [count/sum(self.n_topic_word[k]) for count in self.n_topic_word[k]]
            top_word_ids = np.asarray(normalized_topic).argsort()[-num_words:][::-1]

            print('words for 1st topic: ')
            print([self.id2word[id] for id in top_word_ids])


if __name__ == '__main__':

    dataset = datamodule(num_docs=500)

    LDAmodel = LDA(dataset=dataset,
                   num_topics=25,
                   alpha=1,
                   beta=1)

    LDAmodel.train(iterations=500,
                   eval_every=25,
                   num_words =20)


