import sys
import pickle
import logging
import numpy as np
import pandas as pd
import sklearn.cross_validation
import textcnn as tc

def train_cnn(data_filename, vectors_filename, vectors_cached_filename):
    max_sentence_length = 100
    max_doc_length = 32
    (vectors, X, Y) = tc.data.prepare_data(
            pickle.load(open(data_filename, 'rb')),
            vectors_filename, vectors_cached_filename,
            max_sentence_length, max_doc_length)

    (nsents, nwords) = np.shape(X[0])
    logging.info('# sentences: %d' % nsents)
    logging.info('# words: %d' % nwords)
    (nsamples, _) = np.shape(Y)
    logging.info('Number of samples: %d' % nsamples)

    folder = sklearn.cross_validation.StratifiedKFold(Y[:,0], n_folds=5, random_state=42)
    (train_idx, val_idx) = [(t, v) for (t, v) in folder][0]

    an = tc.cnn.TextCNN(nsents, nwords, vectors,
            train_batch_size=32, test_batch_size=32, n_epochs=5,
            filter_sizes=[3,4,5], num_filters=128, l2_reg_lambda=0.2,
            learning_rate=0.0002, dropout_prob=0.5, eval_X=X[val_idx], eval_Y=Y[val_idx], eval_iter=500)
    results = an.fit(X[train_idx], Y[train_idx])
    pickle.dump(results, open('cnn_fit_results.pkl', 'wb'))

def main(args):
    if len(args) != 4:
        print("Usage: train_cnn.py [in/yelp_training_set_review.pkl] [in/GoogleNews-vectors-negative300.bin] [in/GoogleNews.pkl]")
        sys.exit(1)

    data_filename = sys.argv[1]
    vectors_filename = sys.argv[2]
    vectors_cached_filename = sys.argv[3]

    train_cnn(data_filename, vectors_filename, vectors_cached_filename)

if __name__ == "__main__":
    main(sys.argv)
