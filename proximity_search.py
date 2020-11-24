import ast
import math
import time
import re

from index.constants import *
from index.merge_index import get_param
import json

def get_span(pos_list, word_pos, exception_list=None):
    min = float('inf')
    max = float('-inf')
    min_pos = None
    if exception_list:
        for pos in exception_list:
            if pos in pos_list:
                pos_list.remove(pos)
    for pos in pos_list:
        term = word_pos[pos[0]][pos[1]]
        if term < min:
            min = word_pos[pos[0]][pos[1]]
            min_pos = (pos[0], pos[1])
        if term > max:
            max = word_pos[pos[0]][pos[1]]
    span = max - min
    return span, min_pos


def clip_pos(pos, word_pos):
    len_row = len(word_pos[pos[0]])
    new_pos = None
    if pos[1] >= len_row:
        new_pos = (pos[0], pos[1] - 1)
    if pos[0] >= len(word_pos):
        new_pos = (pos[0] - 1, pos[1])
    if pos[1] < len_row and pos[0] < len(word_pos):
        new_pos = pos
    return word_pos[new_pos[0]][new_pos[1]]


def get_pos_row(word_pos, pos_list):
    value = [clip_pos(pos, word_pos) for pos in pos_list]
    return value


def get_min_span(word_pos):
    row = 0
    col = len(word_pos)
    row = [(c, row) for c in range(0, col)]
    span_list = []
    min_list = set()
    while True:
        span, min_pos = get_span(row, word_pos, min_list)
        span_list.append(span)
        r = word_pos[min_pos[0]]
        l = len(r) - 1
        m = min_pos[1]
        if min_pos[1] >= l:
            min_list.add(min_pos)
            break
        row.remove(min_pos)
        row.append((min_pos[0], min_pos[1] + 1))
    return min(span_list)


def get_score(span, length):
    alpha = ALPHA
    score = span * length
    proximity_score = math.log(alpha + math.exp(-1 * score))
    return proximity_score


def query_word_positions(doc_id, word_data_list, inverted, catalog, doc_index):
    query_position_matrix = []
    if len(word_data_list) == 1:
        return get_score(0, 1)
    for word in word_data_list:
        term_vector = get_param(word, catalog, inverted)
        term_vector_1 = re.sub(':', '":', term_vector)
        term_vector_3 = re.sub('],', '],"', term_vector_1)
        term_vector_4 = '{{ "{} }}'.format(term_vector_3)
        term_vector_dic = json.loads(term_vector_4)
        index_of_doc_id = doc_index[doc_id]

        if str(index_of_doc_id) in term_vector_dic.keys():
            position_list = term_vector_dic[str(index_of_doc_id)]
            query_position_matrix.append(position_list)
            # query_position_dic.update({word: sorted(position_list)})
    span = get_min_span(query_position_matrix)
    score = get_score(span, len(word_data_list))
    return score

