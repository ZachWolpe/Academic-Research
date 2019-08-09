


"""
Author:
    Zach Wolpe
    zachcolinwolpe@gmail.com
    zachwolpe.com
"""



import sys, os
# unfortunately I was not able to run it on GPU due to overflow problems
import theano

from collections import OrderedDict
from copy import deepcopy
import numpy as np
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
import seaborn as sns
from theano import shared
import theano.tensor as tt
from theano.sandbox.rng_mrg import MRG_RandomStreams

import pymc3 as pm
from pymc3 import math as pmmath
from pymc3 import Dirichlet
from pymc3.distributions.transforms import t_stick_breaking
plt.style.use('seaborn-darkgrid')



# Gensim & collectives
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
import theano
from time import time
import re
from pprint import pprint

# nltk
import nltk
from nltk.corpus import stopwords

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy


class LearnParameters:
    """
    A series of functions to compute LDA leveraging:
        - PyMC3 variational autoencoders
        - SKlearn
        - Gensim Mallet 
        
    """
    
    ######################################################## PYMC3 #######################################################
    ######################################################## PYMC3 #######################################################
    def train_pymc3(docs_te, docs_tr, n_samples_te, n_samples_tr, n_words, n_topics, n_tokens):
        """
        Return: 
            Pymc3 LDA results
        
        Parameters:
            docs_tr: training documents (processed)
            docs_te: testing documents (processed)
            n_samples_te: number of testing docs
            n_samples_tr: number of training docs
            n_words: size of vocabulary
            n_topics: number of topics to learn
            n_tokens: number of non-zero datapoints in processed training tf matrix
            
        """
        
        # Log-likelihood of documents for LDA
        def logp_lda_doc(beta, theta):
            
            """
            Returns the log-likelihood function for given documents.

            K : number of topics in the model
            V : number of words (size of vocabulary)
            D : number of documents (in a mini-batch)

            Parameters
            ----------
            beta : tensor (K x V)
              Word distribution.
            theta : tensor (D x K)
              Topic distributions for the documents.
            """

            def ll_docs_f(docs):
                dixs, vixs = docs.nonzero()
                vfreqs = docs[dixs, vixs]
                ll_docs = vfreqs * pmmath.logsumexp(
                      tt.log(theta[dixs]) + tt.log(beta.T[vixs]), axis=1).ravel()

                # Per-word log-likelihood times no. of tokens in the whole dataset
                return tt.sum(ll_docs) / (tt.sum(vfreqs)+1e-9) * n_tokens

            return ll_docs_f
       
        # fit the pymc3 LDA

        # we have sparse dataset. It's better to have dence batch so that all words accure there
        minibatch_size = 128

        # defining minibatch
        doc_t_minibatch = pm.Minibatch(docs_tr.toarray(), minibatch_size)
        doc_t = shared(docs_tr.toarray()[:minibatch_size])

        with pm.Model() as model:
            theta = Dirichlet('theta', a=pm.floatX((1.0 / n_topics) * np.ones((minibatch_size, n_topics))),
                           shape=(minibatch_size, n_topics), transform=t_stick_breaking(1e-9),
                           # do not forget scaling
                           total_size = n_samples_tr)
            beta = Dirichlet('beta', a=pm.floatX((1.0 / n_topics) * np.ones((n_topics, n_words))),
                         shape=(n_topics, n_words), transform=t_stick_breaking(1e-9))
            # Note, that we defined likelihood with scaling, so here we need no additional `total_size` kwarg
            doc = pm.DensityDist('doc', logp_lda_doc(beta, theta), observed=doc_t)        
      
    
        # Encoder
        class LDAEncoder:
            """Encode (term-frequency) document vectors to variational means and (log-transformed) stds.
            """
            def __init__(self, n_words, n_hidden, n_topics, p_corruption=0, random_seed=1):
                rng = np.random.RandomState(random_seed)
                self.n_words = n_words
                self.n_hidden = n_hidden
                self.n_topics = n_topics
                self.w0 = shared(0.01 * rng.randn(n_words, n_hidden).ravel(), name='w0')
                self.b0 = shared(0.01 * rng.randn(n_hidden), name='b0')
                self.w1 = shared(0.01 * rng.randn(n_hidden, 2 * (n_topics - 1)).ravel(), name='w1')
                self.b1 = shared(0.01 * rng.randn(2 * (n_topics - 1)), name='b1')
                self.rng = MRG_RandomStreams(seed=random_seed)
                self.p_corruption = p_corruption

            def encode(self, xs):
                if 0 < self.p_corruption:
                    dixs, vixs = xs.nonzero()
                    mask = tt.set_subtensor(
                        tt.zeros_like(xs)[dixs, vixs],
                        self.rng.binomial(size=dixs.shape, n=1, p=1-self.p_corruption)
                    )
                    xs_ = xs * mask
                else:
                    xs_ = xs

                w0 = self.w0.reshape((self.n_words, self.n_hidden))
                w1 = self.w1.reshape((self.n_hidden, 2 * (self.n_topics - 1)))
                hs = tt.tanh(xs_.dot(w0) + self.b0)
                zs = hs.dot(w1) + self.b1
                zs_mean = zs[:, :(self.n_topics - 1)]
                zs_rho = zs[:, (self.n_topics - 1):]
                return {'mu': zs_mean, 'rho':zs_rho}

            def get_params(self):
                return [self.w0, self.b0, self.w1, self.b1]
        
            
            
            
            # call Encoder
        encoder = LDAEncoder(n_words=n_words, n_hidden=100, n_topics=n_topics, p_corruption=0.0)
        local_RVs = OrderedDict([(theta, encoder.encode(doc_t))])

        # get parameters 
        encoder_params = encoder.get_params()
        
        
        # Train pymc3 Model
        η = .1
        s = shared(η)
        def reduce_rate(a, h, i):
            s.set_value(η/((i/minibatch_size)+1)**.7)

        with model:
            approx = pm.MeanField(local_rv=local_RVs)
            approx.scale_cost_to_minibatch = False
            inference = pm.KLqp(approx)
        inference.fit(10000, callbacks=[reduce_rate], obj_optimizer=pm.sgd(learning_rate=s),
                      more_obj_params=encoder_params, total_grad_norm_constraint=200,
                      more_replacements={doc_t:doc_t_minibatch})


        # Extracting characteristic words
        doc_t.set_value(docs_tr.toarray())
        samples = pm.sample_approx(approx, draws=100)
        beta_pymc3 = samples['beta'].mean(axis=0)

        
 


    
        # Predictive distribution
        def calc_pp(ws, thetas, beta, wix):
            """
            Parameters
            ----------
            ws: ndarray (N,)
                Number of times the held-out word appeared in N documents.
            thetas: ndarray, shape=(N, K)
                Topic distributions for N documents.
            beta: ndarray, shape=(K, V)
                Word distributions for K topics.
            wix: int
                Index of the held-out word

            Return
            ------
            Log probability of held-out words.
            """
            return ws * np.log(thetas.dot(beta[:, wix]))

        def eval_lda(transform, beta, docs_te, wixs):
            """Evaluate LDA model by log predictive probability.

            Parameters
            ----------
            transform: Python function
                Transform document vectors to posterior mean of topic proportions.
            wixs: iterable of int
                Word indices to be held-out.
            """
            lpss = []
            docs_ = deepcopy(docs_te)
            thetass = []
            wss = []
            total_words = 0
            for wix in wixs:
                ws = docs_te[:, wix].ravel()
                if 0 < ws.sum():
                    # Hold-out
                    docs_[:, wix] = 0

                    # Topic distributions
                    thetas = transform(docs_)

                    # Predictive log probability
                    lpss.append(calc_pp(ws, thetas, beta, wix))

                    docs_[:, wix] = ws
                    thetass.append(thetas)
                    wss.append(ws)
                    total_words += ws.sum()
                else:
                    thetass.append(None)
                    wss.append(None)

            # Log-probability
            lp = np.sum(np.hstack(lpss)) / total_words

            return {
                'lp': lp,
                'thetass': thetass,
                'beta': beta,
                'wss': wss
            }



        inp = tt.matrix(dtype='int64')
        sample_vi_theta = theano.function(
            [inp],
            approx.sample_node(approx.model.theta, 100,  more_replacements={doc_t: inp}).mean(0)
        )
        def transform_pymc3(docs):
            return sample_vi_theta(docs)

        result_pymc3 = eval_lda(transform_pymc3, beta_pymc3, docs_te.toarray(), np.arange(100))
        print('Predictive log prob (pm3) = {}'.format(result_pymc3['lp']))

        
               
        return result_pymc3
        
        

    ###################################################### SKLEARN #######################################################
    ###################################################### SKLEARN #######################################################
    def train_sklearn(docs_te, docs_tr, n_topics):
        """
        Return: SKlearn LDA results
        
        Parameters: 
            docs_te: testing documents (processed)
            docs_tr: training documents (processed)
            n_topics: number of topics to learn
        """

        from sklearn.decomposition import LatentDirichletAllocation

        lda = LatentDirichletAllocation(n_components=n_topics, max_iter=5,
                                        learning_method='online', learning_offset=50.)
        lda.fit(docs_tr)
        sklearn_theta = lda.fit_transform(docs_te)
        beta_sklearn = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
    
    
        result_Sklearn = {
            'lda_model': lda,
            'beta': beta_sklearn,
            'theta': sklearn_theta
            }

        return result_Sklearn

    
    
##################################################### EVALUATE PERFORMANCE ####################################################
##################################################### EVALUATE PERFORMANCE ####################################################



########################################################### POSSIBLE NA ####################################################
########################################################### POSSIBLE NA ####################################################



########################################################### POSSIBLE NA ####################################################
########################################################### POSSIBLE NA ####################################################



    def test_algorithm_performace(sklearn_lda, beta_sklearn, beta_pymc3, docs_te):
        """
        Return:
            CPU time to train the model & Predictive log probability
        """
        
        def eval_lda(transform, beta, docs_te, wixs):
            ## DUPLICATE 
            """Evaluate LDA model by log predictive probability.

            Parameters
            ----------
            transform: Python function
                Transform document vectors to posterior mean of topic proportions.
            wixs: iterable of int
                Word indices to be held-out.
            """
            lpss = []
            docs_ = deepcopy(docs_te)
            thetass = []
            wss = []
            total_words = 0
            for wix in wixs:
                ws = docs_te[:, wix].ravel()
                if 0 < ws.sum():
                    # Hold-out
                    docs_[:, wix] = 0

                    # Topic distributions
                    thetas = transform(docs_)

                    # Predictive log probability
                    lpss.append(calc_pp(ws, thetas, beta, wix))

                    docs_[:, wix] = ws
                    thetass.append(thetas)
                    wss.append(ws)
                    total_words += ws.sum()
                else:
                    thetass.append(None)
                    wss.append(None)

            # Log-probability
            lp = np.sum(np.hstack(lpss)) / total_words

            return {
                'lp': lp,
                'thetass': thetass,
                'beta': beta,
                'wss': wss
            }
        
        
        η = .1
        s = shared(η)
        def reduce_rate(a, h, i):
            s.set_value(η/((i/minibatch_size)+1)**.7)

        with model:
            approx = pm.MeanField(local_rv=local_RVs)
            approx.scale_cost_to_minibatch = False
            inference = pm.KLqp(approx)
        inference.fit(10000, callbacks=[reduce_rate], obj_optimizer=pm.sgd(learning_rate=s),
                      more_obj_params=encoder_params, total_grad_norm_constraint=200,
                      more_replacements={doc_t: doc_t_minibatch})



        
        inp = tt.matrix(dtype='int64')
        sample_vi_theta = theano.function(
            [inp],
            approx.sample_node(approx.model.theta, 100,  more_replacements={doc_t: inp}).mean(0)
        )
        
        def transform_pymc3(docs):
            return sample_vi_theta(docs)
        
   

        ############ PYMC3 ############
        print('Training Pymc3...')
        t0 = time() 
        result_pymc3 = eval_lda(transform_pymc3, beta_pymc3, docs_te.toarray(), np.arange(100))
        pymc3_time = time() - t0
        print('Predictive log prob (pm3) = {}'.format(result_pymc3['lp']))


        ############ sklearn ############
        print('')
        print('')
        print('Training Sklearn...')
        def transform_sklearn(docs):
            thetas = lda.transform(docs)
            return thetas / thetas.sum(axis=1)[:, np.newaxis]

        t0 = time() 
        result_sklearn = eval_lda(transform_sklearn, beta_sklearn, docs_te.toarray(), np.arange(100))
        sklearn_time = time() - t0
        print('Predictive log prob (sklearn) = {}'.format(result_sklearn['lp']))
        
        
        # save the model times
        times = {
            'pymc3 training time': pymc3_time, 
            'sklearn training time': sklearn_time,
        }
        
        return times

