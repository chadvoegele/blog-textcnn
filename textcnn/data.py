import os
import re
import logging
import numpy as np
import pickle
import tempfile
import multiprocessing
import math

def prepare_data(data, vectors_filename, vectors_cached_filename,
        max_sentence_length=None, max_doc_length=None, start=0, end=0.5):

    data = data[math.floor(len(data)*start):math.floor(len(data)*end)]
    with multiprocessing.Pool() as pool:
        data = pool.map(clean_str_map, data)

    if max_sentence_length is None:
        max_sentence_length = max([len(s) for d in data for s in d['text']])

    if max_doc_length is None:
        max_doc_length = max([len(d['text']) for d in data])

    data = [
            {
                'text' : trim_or_pad_sentences(d['text'], max_sentence_length, max_doc_length),
                'stars' : d['stars']
            }
            for d in data
           ]

    all_words = set([w for d in data for s in d['text'] for w in s])

    (wordvec, embedding_size) = load_bin_vec_cached(vectors_filename, vectors_cached_filename)
    wordvec_unk = add_unknown_words(wordvec, embedding_size, all_words)
    word_embeddings = map_wordvec_to_matrix(wordvec_unk, embedding_size, all_words)

    word_index_map = dict([(w,i) for (i,w) in enumerate(all_words)])
    data = [
            {
                'text' : map_sentences_to_indices(d['text'], word_index_map),
                'stars' : d['stars']
            }
            for d in data
           ]
    logging.info('Completed setup data')

    X = np.array([d['text'] for d in data], dtype=np.ndarray)
    Y = np.array([d['stars'] for d in data])
    Y = np.expand_dims(Y, axis=1)

    return (word_embeddings, X, Y)

def clean_str_map(datum):
    cleaned_datum = {
            'text' : [[clean_str(w) for w in s ] for s in datum['text']],
            'stars' : datum['stars']
            }
    return cleaned_datum

def load_bin_vec_cached(vectors_filename, vectors_cached_filename):
    if os.path.exists(vectors_cached_filename):
        logging.info('Loading cached word2vec')
        (wordvec, embedding_size) = pickle.load(open(vectors_cached_filename, 'rb'))
    else:
        (wordvec, embedding_size) = load_bin_vec(vectors_filename)
        pickle.dump((wordvec, embedding_size), open(vectors_cached_filename, 'wb'))
    return (wordvec, embedding_size)

def load_bin_vec(fname):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    logging.info('Loading word2vec from %s' % fname)
    word_vecs = {}
    words_loaded = 0
    with open(fname, 'rb') as f:
        header = f.readline()
        vocab_size, layer1_size = list(map(int, header.split()))
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = chr(int.from_bytes(f.read(1), byteorder='big'))
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            words_loaded = words_loaded + 1
            if words_loaded % 10000 == 0:
                logging.info('%d words loaded' % words_loaded)
    return (word_vecs, layer1_size)

def random_word_vec(k=300):
    """
    From Yoon Kim's implementation.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    return np.random.uniform(-0.25,0.25,k)

def add_unknown_words(wordvec, embedding_size, all_words):
    words_added = 0
    for word in all_words:
        if word not in wordvec:
            if word == '<PAD/>':
                wordvec['<PAD/>'] = np.zeros((embedding_size))
            else:
                wordvec[word] = random_word_vec()
                words_added = words_added + 1
    logging.info('%d/%d=%6.4f words added' % (words_added, len(all_words), words_added/len(all_words)))
    return wordvec

def map_sentences_to_indices(sentences, word_index_map):
    mapped_sentences = [[word_index_map[w] for w in s] for s in sentences]
    mapped_sentences = np.array(mapped_sentences, dtype=np.int)
    return mapped_sentences

def map_wordvec_to_matrix(wordvec, embedding_size, all_words):
    word_index_map = dict([(w,i) for (i,w) in enumerate(all_words)])

    word_embeddings = np.zeros((len(word_index_map), embedding_size), dtype=np.float32)
    for word in all_words:
        word_embeddings[word_index_map[word],:] = wordvec[word]
    return word_embeddings

"""
From Denny Britz
"""
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r'[^A-Za-z0-9(),!?\'\`]', ' ', string)
    string = re.sub(r'\'s', ' \'s', string)
    string = re.sub(r'\'ve', ' \'ve', string)
    string = re.sub(r'n\'t', ' n\'t', string)
    string = re.sub(r'\'re', ' \'re', string)
    string = re.sub(r'\'d', ' \'d', string)
    string = re.sub(r'\'m', ' \'m', string)
    string = re.sub(r'\'ll', ' \'ll', string)
    string = re.sub(r',', ' , ', string)
    string = re.sub(r'!', ' ! ', string)
    string = re.sub(r'\(', ' \( ', string)
    string = re.sub(r'\)', ' \) ', string)
    string = re.sub(r'\?', ' \? ', string)
    string = re.sub(r'\s{2,}', ' ', string)
    return string.strip().lower()

def trim_or_pad_sentences(sentences, length, doc_length, padding_word='<PAD/>'):
    sequence_length = max(len(x) for x in sentences)
    logging.info('Max sentence length is %d' % sequence_length)
    logging.info('Trimming or padding sentences to %d' % length)
    logging.info('Doc length is %d' % len(sentences))
    logging.info('Trimming or padding document to %d' % doc_length)
    n_padded = 0
    n_trimmed = 0
    padded_sentences = []
    for i in range(min(len(sentences), doc_length)):
        sentence = sentences[i]
        sentence_length = len(sentence)
        if (sentence_length <= length):
            num_padding = length - sentence_length
            new_sentence = sentence + [padding_word] * num_padding
            padded_sentences.append(new_sentence)
            n_padded = n_padded + 1
        else:
            padded_sentences.append(sentence[:length])
            n_trimmed = n_trimmed + 1
    logging.info('%d sentences trimmed, %d sentences padded' % (n_trimmed, n_padded))

    padding_sentences_needed = doc_length-len(sentences)
    if padding_sentences_needed > 0:
        padded_sentences.extend([[padding_word]*length]*(padding_sentences_needed))
        logging.info('%d padding sentences added' % padding_sentences_needed)

    return padded_sentences

def pad_sentences(sentences, padding_word='<PAD/>'):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    logging.info('Padding sentences to %d' % sequence_length)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences
