import json
import sys
import os
import multiprocessing
import pickle
from spacy.en import English

nlp = English()

def load_data(data_filename):
    with open(data_filename, 'r') as fopen:
        data = [json.loads(line) for line in fopen.readlines()]
        text_rating = [{'text': d['text'], 'stars': d['stars']} for d in data]
        return text_rating

def split_sentence(text):
    doc = nlp(text)
    sentences = [
            [w for w in s if not (w.is_oov or w.is_space or w.is_punct)]
            for s in doc.sents ]
    sentences_text = [[ w.text for w in s ] for s in sentences]
    return sentences_text

def preprocess_datum(datum):
    preprocessed_datum = {
            'stars': datum['stars'],
            'text': split_sentence(datum['text'])
            }
    return preprocessed_datum

def preprocess_data(data):
    pool = multiprocessing.Pool()
    preprocessed_data = pool.map(preprocess_datum, data)
    return preprocessed_data

def main(args):
    if len(args) != 3:
        print("Usage: preprocess_data.py [in/yelp_training_set_review.json] [out/yelp_training_set_review.pkl]")
        sys.exit(1)

    data_filename = sys.argv[1]
    data = load_data(data_filename)
    preprocessed_data = preprocess_data(data)
    pkl_name = sys.argv[2]
    pickle.dump(preprocessed_data, open(pkl_name, 'wb'))

if __name__ == "__main__":
    main(sys.argv)
