import pickle
from tqdm import tqdm
import traceback
from pathlib import Path
import re
from index.constants import *
import ast

from index.constants import QUERY_VECTOR_FILE
from index.merge_index import get_param
from index.create_inverted_docs import get_punc_removed_word, get_stemmed, process_text


def remove_dashes(word_list):
    dash = '-'
    for each_word in word_list:
        if dash in each_word:
            list_trunc_words = each_word.split(dash)
            word_list.remove(each_word)
            for words in list_trunc_words:
                word_list.append(words)
    return word_list


def get_processed_query_text(query_text):
    custom_stopwords_for_query = ["document", "include", "report", "discuss", "cite", "describe", "predict", "identify",
                                  "will"]
    # query_text = analyze_query(query_text)
    stopwords = open(PATH_TO_STOPWORDS, "r").read().split("\n") + custom_stopwords_for_query
    word_list = query_text.split()
    new_word_list = []
    for key_word in word_list:
        new_word_list.append(key_word) if key_word not in stopwords else None
    new_word_list, _ = process_text(' '.join(new_word_list))
    stemmed = get_stemmed(new_word_list)
    return stemmed
    pass


def get_query_words(query_common_word_dic=None):
    query_unique_words = []
    query_dic = {}
    with open(PATH_TO_Query) as query_file:
        for query in query_file.readlines():
            query_text = re.findall(PATTERN_QUERY, query)
            query_num = re.findall(PATTERN_QUERY_NO, query)
            if len(query_num) == 1:
                query_num = query_num[0]
            if len(query_text) == 1:
                query_text = query_text[0]
                if query_common_word_dic:
                    query_text += ' {}'.format(' '.join(query_common_word_dic[query_num]))
                query_text = get_punc_removed_word(query_text)
                query_list = get_processed_query_text(query_text)
            query_unique_words += query_list
            query_dic[query_num] = query_list
    return list(set(query_unique_words)), query_dic


def write_to_file(text):
    # def save_data(data, file_path):
    with open(QUERY_VECTOR_FILE, 'wb') as file:
        pickle.dump(text, file)
    print("Saving the data to the text file query_term_vectors.txt")


def read_data(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def get_list_doc_ids():
    with open(PATH_TO_DOCLIST) as doc_list_file:
        doc_ids = re.findall(PATTERN_DOCID, doc_list_file.read())
    return doc_ids


def add_data_to_esdata(query_words_esdata, word, doc_id, doc_freq, ttf, term_freq, doc_len):
    if word not in query_words_esdata.keys():
        if term_freq == 0:
            body = None
        else:
            body = {
                'doc_freq': doc_freq,
                'ttf': ttf,
                'term_freq': term_freq,
                'doc_len': doc_len
            }
        query_words_esdata[word] = {
            doc_id: body
        }
    else:
        if doc_id not in query_words_esdata[word].keys():
            if term_freq == 0:
                body = None
            else:
                body = {
                    'doc_freq': doc_freq,
                    'ttf': ttf,
                    'term_freq': term_freq,
                    'doc_len': doc_len
                }
            query_words_esdata[word][doc_id] = body
        else:
            print("Word already exists in query_words_esdata")
    return query_words_esdata
    pass


def get_doc_len(doc_id, term_vector):
    if doc_id in DOC_LEN.keys():
        return DOC_LEN[doc_id]
    else:
        terms = term_vector['term_vectors']['text']['terms']
        total_words_in_doc = 0
        for keys in terms.keys():
            total_words_in_doc += terms[keys]['term_freq']
        DOC_LEN[doc_id] = total_words_in_doc
        return total_words_in_doc
    pass


def get_term_freq(doc_id, param_dic, doc_index):
    key = [key in param_dic for key, value in doc_index.items() if value['doc_no'] == doc_id][0]
    return param_dic[key[0]]
    pass


def generate_query_words_parameters(query_common_word_dic=None):
    if query_common_word_dic is None:
        query_common_word_dic = {}
    query_words_esdata = {}
    query_words, _ = get_query_words(query_common_word_dic)
    doc_ids = get_list_doc_ids()
    count = 0
    doc_index = read_data(PATH_TO_DOC_INDEX)
    print("Getting, term_freq, doc_freq, ttf for all words from the query corresponding to each document")
    inverted = open(PATH_TO_MERGED_INVERTED / 'inverted_doc_L7_F1')
    catalog = open(PATH_TO_MERGED_CATALOG / 'merged_catalog_L7_F1')
    catalog_read = catalog.read()
    try:
        for word in query_words:
            regex = r'\n{}.*\n'.format('{}:'.format(word))
            catalog.seek(0)
            params = get_param(word, catalog_read, inverted)
            param_dic = ast.literal_eval('{{ {} }}'.format(params))
            ttf = sum([len(position) for position in param_dic.values()])
            for key, value in param_dic.items():
                doc_id = doc_index[key]['doc_no']
                doc_len = doc_index[key]['doc_len']
                doc_freq = len(param_dic.keys())
                term_freq = len(param_dic[key])

                query_words_esdata = add_data_to_esdata(query_words_esdata, word, doc_id, doc_freq, ttf,
                                                        term_freq, doc_len)
            print('Query vector generated for term {}'.format(word))
    except Exception as e:
        print(traceback.format_exc())
        print('Could not get the details from mtermvectors api: {}'.format(e))


    print("Successfully fetched for all details")
    print("--Validating the vectors")
    validate_query_vectors(query_words_esdata)
    print("--Saving query vector in file")
    write_to_file(query_words_esdata)

    return query_words_esdata


def validate_query_vectors(query_vectors_dic):
    fault_list = []
    query_word_list, _ = get_query_words()
    for word in query_word_list:
        word_found = False
        for same_word in [k for k in query_vectors_dic.keys()]:
            if same_word == word:
                word_found = True
        if not word_found:
            fault_list.append(word)
    if not len(fault_list):
        print('Existing query term vectors have been validated. ')
        return query_vectors_dic


def generate_query_vectors(query_to_common_word_list=None):
    vectors = {}

    # Try to read the file and check if query vector is already saved
    if not query_to_common_word_list:
        try:
            print('Trying to read the query vectors from the file: {}'.format(QUERY_VECTOR_FILE))
            vectors = read_data(QUERY_VECTOR_FILE)
        except FileNotFoundError:
            print('Vector file does not exists, generating, validating and saving them in a file now')
            vectors = generate_query_words_parameters(query_to_common_word_list)
            return vectors

        # If file does not exist, generate the query vectors, validate them and save them in a file
        if len(vectors) == 0:
            print('Could not find vectors in the file, Generating, validating and saving them in a file now')
            vectors = generate_query_words_parameters(query_to_common_word_list)
            return vectors
        else:
            # If file exists, return the vector
            print('Query vectors already exists.')
            return vectors
    else:
        print('Generating query vector for models with Pseudo relevance.')
        vectors = generate_query_words_parameters(query_to_common_word_list)
        return vectors


def get_query_vectors(query_to_common_word_list=None):
    return generate_query_vectors(query_to_common_word_list)


generate_query_words_parameters()
