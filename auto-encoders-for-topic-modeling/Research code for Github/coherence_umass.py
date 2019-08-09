
"""
Author:
    Zach Wolpe
    zachcolinwolpe@gmail.com
    zachwolpe.com
"""



import numpy as np
from math import log10


class coherence_umass:
    """
    Calculate the model coherence leveraging the UMASS technique
    """


    def coherence(corpus, beta, feature_names, n_common_words=10, epsilon=1, print_=False):
        """
        Return:
            Topic Model Coherence

        Looking at only a one lag 

        Parameters:
            corpus: corpus of documents
            topic_word: 
            beta: learnt Beta Distribution
            feature_names: feature names of tokens
            n_words: number of N most common words to return
            epsilon: smoothing parameter
            print_: if True, print coherence score

        """
        beta = np.array(beta)
        corpus = np.array(corpus)
        n_topics = beta.shape[0] # number of topics

        common_words = [] # find n_words most common words per topic
        for i in range(len(beta)):
                common_words.append([feature_names[j] for j in beta[i].argsort()[:-n_common_words - 1:-1]])



        coherence = []
        for k in range(n_topics):
            # for each topic


            for vi in range(n_common_words-1):
                # for each word in each topic
                D_vi = 0


                for doc in corpus:
                    # for each document 

                    D_vi_vj = 0
                    vj = vi+1

                    if common_words[k][vi] in doc:
                        # word is in the document
                        D_vi = D_vi + 1

                        if common_words[k][vj] in doc:
                            # only check lag-1 words if word vi in doc
                            D_vi_vj = D_vi_vj + 1

                if D_vi != 0:
                    # catch errors
                    coherence.append(log10((D_vi_vj+epsilon)/D_vi))

        model_coherence = np.sum(coherence)/n_topics

        if print_: print("coherence for each topic: ", model_coherence)

        return model_coherence
        