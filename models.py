import ast
import math
import re
import time

from tqdm import tqdm

from index.query_processing import get_query_words, get_query_vectors
from index.merge_index import get_vocab_size
from index.create_inverted_docs import create_index
from index.constants import *
from index.proximity_search import *
from index.merge_index import read_data
from elasticsearch import Elasticsearch

ES = Elasticsearch()

def generate_result_matrix(results_dic, query_no, doc_no, score_query):
    if query_no not in results_dic.keys():
        results_dic[query_no] = {
            doc_no: score_query
        }
    else:
        if doc_no not in results_dic[query_no]:
            results_dic[query_no][doc_no] = score_query
    return results_dic


def write_result(result_dic, filename):
    with open(filename, 'w') as f:
        for query_no in result_dic.keys():
            rank = 1
            count = 0
            for doc_no in sorted(result_dic[query_no], key=result_dic[query_no].get, reverse=True)[:1000]:
                text = '{} Q0 {} {} {} Exp\n'.format(query_no, doc_no, rank, result_dic[query_no][doc_no])
                rank += 1
                count += 1
                f.write(text)
        print(count)


def get_doc_list():
    with open(PATH_TO_DOCLIST) as file:
        doc_no_list = re.findall(PATTERN_DOCID, file.read())
    return doc_no_list


def get_term_details(term_dic):
    if not term_dic:
        doc_freq_word = 0
        ttf_word = 0
        term_freq_word = 0
        doc_len_word = 0
        return doc_freq_word, ttf_word, term_freq_word, doc_len_word
    else:
        doc_freq_word = term_dic['doc_freq']
        ttf_word = term_dic['ttf']
        term_freq_word = term_dic['term_freq']
        doc_len_word = term_dic['doc_len']
        return doc_freq_word, ttf_word, term_freq_word, doc_len_word


def get_okapi_score(data_word_list, mean_doc_len):
    score = 0
    for data_of_words in data_word_list:
        doc_freq_word, ttf_word, term_freq_word, doc_len_word = get_term_details(data_of_words)
        okapi_tf_for_word_doc = term_freq_word / (term_freq_word + 0.5 + (1.5 * (doc_len_word / mean_doc_len)))
        score += okapi_tf_for_word_doc
    return score


def get_tf_idf_score(data_word_list, mean_doc_len):
    score = 0
    for data_of_words in data_word_list:
        doc_freq_word, ttf_word, term_freq_word, doc_len_word = get_term_details(data_of_words)
        first_term = (term_freq_word / (term_freq_word + 0.5 + (1.5 * (doc_len_word / mean_doc_len))))
        if first_term == 0:
            tf_idf_for_word_doc = 0
        else:
            second_term = (math.log(NUMBER_OF_DOCUMENTS / doc_freq_word))
            tf_idf_for_word_doc = first_term * second_term

        score += tf_idf_for_word_doc
    return score


def get_okapi_bm25_score(data_word_list, mean_doc_len):
    score = 0
    k1 = .617
    k2 = 1.114617
    b = .941
    for data_of_words in data_word_list:
        doc_freq_word, ttf_word, term_freq_word, doc_len_word = get_term_details(data_of_words)
        first_term = math.log((NUMBER_OF_DOCUMENTS + 0.5) / (doc_freq_word + 0.5))

        second_term = (term_freq_word + (k1 * term_freq_word)) / \
                      (term_freq_word + k1 * ((1 - b) + b * (doc_len_word / mean_doc_len)))

        third_term = (term_freq_word + (k2 * term_freq_word)) / (term_freq_word + k2)

        okapi_bm25_for_word_doc = first_term * second_term * third_term
        score += okapi_bm25_for_word_doc
    return score


def get_unigram_lm_laplace_score(data_word_list, vocab):
    score = 0
    for data_of_words in data_word_list:
        doc_freq_word, ttf_word, term_freq_word, doc_len_word = get_term_details(data_of_words)

        unigram_lm_laplace_for_word_doc = (term_freq_word + 1) / (doc_len_word + 4000)

        score += math.log10(unigram_lm_laplace_for_word_doc)
    return score
    pass


def get_unigram_lm_jm_score(data_word_list, vocab_size):
    score = 0
    lamda = 0.6
    if not data_word_list:
        score += math.log(0 + (1 - lamda) * .00001)
    for data_of_words in data_word_list:
        doc_freq_word, ttf_word, term_freq_word, doc_len_word = get_term_details(data_of_words)

        if term_freq_word != 0:
            first_term = lamda * (term_freq_word / doc_len_word)
            second_term = (1 - lamda) * ((ttf_word - term_freq_word) / (SUM_TTF_CORPUS - doc_len_word))
        else:
            first_term = 0
            second_term = (1 - lamda) * .00001
        score += math.log(first_term + second_term)
    return score


def create_score_file_for_ES(result_es, query_dic):
    for query_no in query_dic.keys():
        size = 5000
        body = {
            "size": size,
            "query": {
                "match": {
                    "text": " "
                }
            }
        }
        text_format = " ".join(query_dic[query_no])
        body['query']['match']['text'] = text_format
        result = ES.search(body, scroll='2m')
        scrollId = result['_scroll_id']
        hit_list = result['hits']['hits']
        while size < 84678:
            for hit in hit_list:
                doc_no = hit['_id']
                score_no = hit['_score']
                dic = {query_no: {doc_no: hit['_score']}}
                result_es = generate_result_matrix(result_es, query_no, doc_no, score_no)
            result = ES.scroll(scroll_id=scrollId, scroll='2m')
            hit_list = result['hits']['hits']
            size += 5000
    return result_es


def get_doc_statistic():

    avg = SUM_TTF_CORPUS / NUMBER_OF_DOCUMENTS

    return get_vocab_size(), avg


def run_models(run, query_to_common_word_list=None):
    # Creating Index
    try:
        inverted_file = open(PATH_TO_MERGED_INVERTED / 'inverted_doc_L7_F1')
        catalog_file = open(PATH_TO_MERGED_CATALOG / 'merged_catalog_L7_F1').read()
    except Exception:
        print("Merged catalog or inverted file not found. Please delete all catalogs and inverted docs before "
              "proceeding")
        create_index()

    # Creating Query vector: A mapping of all unique words in all query to their corresponding details in ES
    if query_to_common_word_list:
        query_words_data = get_query_vectors(query_to_common_word_list)
    else:
        query_words_data = get_query_vectors()
    # Getting a dictionary of query no to the corresponding word list(processed)
    _, query_dic = get_query_words(query_to_common_word_list)

    # Fetching Document statistic
    print('Fetching document statistic')
    vocab_size, mean_doc_len = get_doc_statistic()
    print('---Average document length: {}'.format(mean_doc_len))
    print('---Vocab size: {}'.format(vocab_size))

    print('Fetching list of all documents')
    list_of_document_id = get_doc_list()
    print('---List fetched')

    result_elasticsearch = {}
    results_okapi_tf = {}
    results_tf_idf = {}
    results_okapi_bm25 = {}
    results_unigram_laplace = {}
    results_unigram_jelinek = {}
    proximity = 0
    doc_index_proximity = read_data(PATH_TO_DOC_INDEX_PROXIMITY)

    # print('Generating results using Elasticsearch\'s model, for each document')
    # results_es = create_score_file_for_ES(result_elasticsearch, query_dic)
    # print('---Generating score results.')
    # write_result(results_es, OUTPUT_FILE_ES)
    # print('---Results generated')

    # Running other models for generating score
    print('Running other score models, for each document')
    # Loop through all queries
    for query_num in (query_dic.keys()):
        # Loop through all doc ids
        for doc_id in tqdm(list_of_document_id):
            word_data_list = []
            word_list = []
            # Loop through all the words in the query
            for word in query_dic[query_num]:
                if doc_id in query_words_data[word].keys():
                    word_data = query_words_data[word][doc_id]
                    word_data_list.append(word_data)
                    word_list.append(word)

            # proximity search
            proximity = query_word_positions(doc_id, word_list, inverted_file, catalog_file, doc_index_proximity) \
                if len(word_data_list) > 1 and run_dic['proximity'] else 0
            # VECTOR SPACE MODELS
            # Calculate Okapi TF Score
            if run['okapi_tf']:
                # print("Running Okapi TF")
                okapi_tf_score = get_okapi_score(word_data_list, mean_doc_len) + proximity
                results_okapi_tf = generate_result_matrix(results_okapi_tf, query_num, doc_id, okapi_tf_score)

            # Calculate TF IDF Score
            if run['tf_idf']:
                # print("Running TF IDF")
                tf_idf_score = get_tf_idf_score(word_data_list, mean_doc_len)
                results_tf_idf = generate_result_matrix(results_tf_idf, query_num, doc_id, tf_idf_score)

            # Calculate Okapi BM25 Score
            if run['okapi_bm25']:
                # print("Running Okapi BM25")
                okapi_bm25_score = get_okapi_bm25_score(word_data_list, mean_doc_len) + proximity
                results_okapi_bm25 = generate_result_matrix(results_okapi_bm25, query_num, doc_id, okapi_bm25_score)

            # LANGUAGE SPACE MODELS
            # Calculate score for Unigram  LM with laplace smoothing
            if run['unigram_laplace']:
                # print("Running Unigram laplace")
                unigram_lm_laplace_score = get_unigram_lm_laplace_score(word_data_list, vocab_size)
                results_unigram_laplace = generate_result_matrix(results_unigram_laplace, query_num, doc_id,
                                                                 unigram_lm_laplace_score)

            # Calculate score for Unigram LM with  Jelinek-Mercer smoothing
            if run['unigram_jelinek']:
                # print("Running Unigram jelinek")
                unigram_lm_jelinek_score = get_unigram_lm_jm_score(word_data_list, vocab_size)
                results_unigram_jelinek = generate_result_matrix(results_unigram_jelinek, query_num, doc_id,
                                                                 unigram_lm_jelinek_score)

    print('---Generating score results for other score models.')
    write_result(results_okapi_tf, OUTPUT_FILE_OKAPI_TF) if run['okapi_tf'] else None
    write_result(results_tf_idf, OUTPUT_FILE_TF_IDF) if run['tf_idf'] else None
    write_result(results_okapi_bm25, OUTPUT_FILE_OKAPI_BM25) if run['okapi_bm25'] else None
    write_result(results_unigram_laplace, OUTPUT_FILE_UNI_LAP) if run['unigram_laplace'] else None
    write_result(results_unigram_jelinek, OUTPUT_FILE_UNI_JEL) if run['unigram_jelinek'] else None
    print('---Results generated')


def main(run_dict, query_to_common_word_list=None):
    run_models(run_dict, query_to_common_word_list)


if __name__ == "__main__":
    run_dic = {
        'unigram_jelinek': False,
        'unigram_laplace': False,
        'okapi_bm25': True,
        'tf_idf': True,
        'okapi_tf': True,
        'proximity': False
    }
    main(run_dic)
