import logging
import math
import pickle
import sys
import multiprocessing
import textcnn as tc
import numpy as np
import sklearn.cross_validation
import sklearn.linear_model

def train_ngram(data_filename):
    data = load_data(data_filename)
    feature_words = find_feature_words(data)
    (X, Y) = build_data_vectors(data, feature_words)

    (nreviews, feature_dim) = np.shape(X)
    logging.info('# reviews: %d' % nreviews)
    logging.info('feature dimension: %d' % feature_dim)

    folder = sklearn.cross_validation.StratifiedKFold(Y, n_folds=5, random_state=42)
    (train_idx, val_idx) = [(t, v) for (t, v) in folder][0]

    regression = sklearn.linear_model.LinearRegression()
    regression.fit(X[train_idx,:], Y[train_idx])

    train_error = calculate_error(regression, X[train_idx,:], Y[train_idx])
    validation_error = calculate_error(regression, X[val_idx,:], Y[val_idx])

    results = {
            'model': regression,
            'train_error': train_error,
            'validation_error': validation_error
            }
    logging.info('train_error: %f', train_error)
    logging.info('validation_error: %f', validation_error)
    pickle.dump(results, open('ngram_fit_results.pkl', 'wb'))

def calculate_error(model, X, Y):
    (nsamples,) = np.shape(Y)
    predY = model.predict(X)
    losses = (Y-predY)**2
    sqrt_mean_loss = math.sqrt(sum(losses)/nsamples)
    return sqrt_mean_loss

def load_data(data_filename, start=0, end=0.5):
    data = pickle.load(open(data_filename, 'rb'))
    data = data[math.floor(len(data)*start):math.floor(len(data)*end)]
    with multiprocessing.Pool() as pool:
        data = pool.map(tc.data.clean_str_map, data)
    return data

def find_feature_words(data):
    # Find top 500 so that:
    #   - have at least 10 after filtering non unique
    #   - capture lower frequency words
    top_word_counts = [
            tc.majority.majority_elements(words_by_star, 500)
            for s in set((d['stars'] for d in data))
            for data_by_star in [(d for d in data if d['stars']==s)]
            for words_by_star in [(w for d in data_by_star for l in d['text'] for w in l)]
        ]

    top_words = (w for s in top_word_counts for w in s.keys())
    top_words_unique_counts = count_words(top_words)

    # A word is not unique is it appears in reviews with 4 or more stars.
    non_unique_threshold = 4
    non_unique_words = [w for (w, c) in top_words_unique_counts.items() if c >= non_unique_threshold]

    words_per_star = 10
    unique_top_words = [
            [w for (w, c) in sorted_words if w not in non_unique_words][:words_per_star]
            for top_word_counts_by_star in top_word_counts
            for sorted_words in [sorted(top_word_counts_by_star.items(), key=lambda x: -x[1])]
            ]
    feature_word_set = list(set((w for s in unique_top_words for w in s)))

    return feature_word_set

def build_data_vectors(data, feature_words):
    prepared_data = [
            {
                'X': build_feature_vector(d['text'], feature_words),
                'Y': d['stars']
            }
            for d in data
            ]
    X = np.array([d['X'] for d in prepared_data])
    Y = np.array([d['Y'] for d in prepared_data])
    return (X, Y)

def build_feature_vector(text, feature_word_set):
    feature_counts = count_words([w for l in text for w in l], feature_word_set)
    feature_vector = [
            feature_counts[w] if w in feature_counts else 0
            for w in feature_word_set
            ]
    return feature_vector

def count_words(stream, word_set=None):
    counts = {}
    for word in stream:
        if word_set is not None and word not in word_set:
            continue

        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts

def main(args):
    if len(args) != 2:
        print("Usage: train_ngram.py [in/yelp_training_set_review.pkl]")
        sys.exit(1)

    data_filename = sys.argv[1]

    train_ngram(data_filename)

if __name__ == "__main__":
    main(sys.argv)
