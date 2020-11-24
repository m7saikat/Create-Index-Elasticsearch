import re

from tqdm import tqdm
import pickle
import random

import time
import string
import nltk
import zlib

from nltk.stem import PorterStemmer


ps = PorterStemmer()

from index.constants import *

translator = str.maketrans('', '', string.punctuation)


def setup(root=None):
    root = ROOT
    path_to_documents = root / PATH_TO_AP_DATASET
    abs_path_to_stopwords = root / PATH_TO_STOPWORDS_CREATE_INDEX
    return path_to_documents, abs_path_to_stopwords
    pass


def get_punc_removed_word(text):
    text_new = text.translate(translator)
    text_new_tokenized = nltk.word_tokenize(text_new)
    words = [word.lower() for word in text_new_tokenized if word.isalpha()]
    return ' '.join(words)


def process_text(query_text):
    word_list = [word.lower() for word in query_text.split()]
    stopwords = open(PATH_TO_STOPWORDS, "r").read().split("\n")
    processed_list = []
    for key_word in word_list:
        processed_list.append(key_word.lower()) if key_word not in stopwords else None
    return processed_list, word_list
    pass


def get_stemmed(processed_list):
    stemmed = []
    for word in processed_list:
        stemmed.append(ps.stem(word))
    return stemmed


def tokenize_words(words_index, doc_no, texts):
    no_punc_text = get_punc_removed_word(texts)
    processed_list, raw_word_list = process_text(no_punc_text)
    processed_list = get_stemmed(processed_list)
    doc_len = len(raw_word_list)
    for word in processed_list:

        position_list = [i + 1 for i, x in enumerate(processed_list) if x == word]
        if word not in words_index.keys():
            words_index.update({word: {doc_no: position_list}})
        elif doc_no not in words_index[word].keys():
            words_index[word].update({doc_no: position_list})
        elif word in words_index.keys() \
                and doc_no in words_index[word].keys() \
                and position_list == words_index[word][doc_no]:
            continue
        else:
            print('Invalid word entry')
    return words_index, doc_len


def write_inverted_doc(word, cursor, param, inverted_file, catalog_file, compress=False):
    start = inverted_file.seek(cursor)

    param = "".join(str(param).split()).replace('{', '').replace('}', '').replace('(', '').replace(')', '')
    param = zlib.compress(b'param') if compress else param
    inverted_file.write('{}'.format(param))
    cursor = inverted_file.tell()
    catalog_file.write('{}: {} {}\n'.format(word, start, cursor - 1))
    return cursor


def get_catalog_inverted_files(doc_count):
    filename = 'inverted_doc_{}'.format(doc_count)
    catalog_file_name = 'catalog_{}'.format(doc_count)

    inverted_file = open(PATH_TO_INVERTED / filename, 'a')
    catalog_file = open(PATH_TO_CATALOG / catalog_file_name, 'a')
    catalog_file.write('\n')
    return catalog_file, inverted_file


def write_to_file(filename, text):
    # def save_data(data, file_path):
    with open(filename, 'wb') as file:
        pickle.dump(text, file)


def create_index():
    print('Generating document id to text map ...')
    documents_path, stop_word_file = setup()
    # print('Reading data fr  om the location : {}'.format(str(documents_path)))
    count = 0
    doc_count = 1

    proximity_mapping = {}
    doc_no_index_dict = {}
    words_vector = {}
    doc_no_index = 0
    start_time = time.time()
    sum_ttf = 0
    for document in documents_path.iterdir():
        file = open(document, 'r')
        doc = re.finditer(PATTERN_DOC, file.read(), re.DOTALL)
        # Loop through all documents in a file
        for match_group in doc:
            # Get doc no
            doc_no = re.findall(PATTERN_DOC_NO, match_group.group(), re.DOTALL)[0]

            # Get texts
            texts = re.findall(PATTERN_TEXT, match_group.group(), re.DOTALL)[0]

            # create a dictionary of the form 'token' : {'doc_no': [<Position of token>]}
            words_vector, doc_len = tokenize_words(words_vector, doc_no_index, texts)

            no_punc_text = get_punc_removed_word(texts)
            _, raw_word_list = process_text(no_punc_text)
            doc_len = len(raw_word_list)
            count += 1

            # Store doc_no and doc_length in a separate dictionary and then write it to a file
            doc_no_index_dict[doc_no_index] = {
                'doc_no': doc_no,
                'doc_len': doc_len
            }

            #Store a reverse mapping for proximiity search
            proximity_mapping.update({doc_no: doc_no_index})
            # print('', end='\r--Documents read: {}/{}'.format(count * doc_count, 85000).format())
            doc_no_index += 1
            # Process 1000 document at once. Create inverted file and the catalog.
            if count % 1000 == 0:
                print('\n----1000 document read Documents, cataloging 1000 document')

                cursor = 0
                catalog_file, inverted_file = get_catalog_inverted_files(doc_count)

                for word in tqdm(words_vector.keys()):
                    cursor = write_inverted_doc(word, cursor, words_vector[word], inverted_file,
                                                catalog_file)
                print("Catalog created")
                words_vector = {}
                count = 0
                doc_count += 1
                print("--- %s seconds ---" % (time.time() - start_time))

    write_to_file(PATH_TO_DOC_INDEX, doc_no_index_dict)
    write_to_file(PATH_TO_DOC_INDEX_PROXIMITY, proximity_mapping)
    print(sum_ttf)


def test_catalog(catalog_file, inverted_file):
    print('Testing catalog: {} against the inverted doc {}'.format(catalog_file, inverted_file))
    lines = catalog_file.readlines()
    a = inverted_file.readlines()
    lines_subset = [random.choice(lines) for i in range(0, 200000)]
    for line in tqdm(lines):
        if line is '\n':
            continue
        entries = line.split(": ")
        word_fom_catalog = entries[0]
        start = int(entries[1].split(' ')[0])
        end = int(entries[1].split(' ')[1].strip())
        inverted_file.seek(start)
        merged_line = inverted_file.read(end - start + 1)
        word_from_inverted_doc = merged_line.split(":")[0]
        length_param = len(merged_line[merged_line.index(word_from_inverted_doc) + len(word_from_inverted_doc) + 1:])
        expected_len_param = end - start - len(word_fom_catalog)
        if word_fom_catalog != word_from_inverted_doc and expected_len_param == length_param:
            print("Error: {} not correctly catalogued".format(word_fom_catalog))
            print(merged_line)


# create_index()
# Randomly verifying 2 catalogs
# test_suite = [random.randint(1, 84) for i in range(0, 3)]
# for doc_num in test_suite:
# test_catalog(open(PATH_TO_CATALOG / 'catalog_{}'.format(1), 'r'), open(PATH_TO_INVERTED / 'inverted_doc_{}'.format(1), 'r'))