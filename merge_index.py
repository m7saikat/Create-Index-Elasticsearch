import ast
import glob
import sys
import zlib

from index.constants import *
from index.create_inverted_docs import write_inverted_doc
from index.create_inverted_docs import test_catalog
import shutil
import pickle
from tqdm import tqdm
import itertools
import time
import re


def read_data(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def write_to_file(filename, text):
    with open(filename, 'wb') as file:
        pickle.dump(text, file)


def get_words_catalog(catalog_line):
    word_list = [line.split(':')[0] for line in catalog_line]
    return sorted(word_list)


def get_param(word, catalog, inverted, compress=False):
    regex = r'\n{}.*\n'.format('{}:'.format(word))
    line = re.findall(regex, catalog)

    if len(line) != 1:
        print (word)
        print(line)
        print("Wrong cataloging")
        sys.exit(1)
    else:
        line = line[0]
    entries = line.split(": ")
    word_fom_catalog = entries[0].strip()
    start = int(entries[1].split(' ')[0])
    end = int(entries[1].split(' ')[1].strip())
    inverted.seek(start)

    param = inverted.read(end - start + 1)
    if compress:
        param = zlib.decompress(param)
    new_param = param.replace('{', '').replace('}', '').replace('{}:'.format(word_fom_catalog), '')
    return new_param


def merge_doc(catalog_1, catalog_2, merged_catalog, inverted_doc_1, inverted_doc_2, merged_inverted_doc, cursor):
    cat_1_text = catalog_1.read()
    catalog_1.seek(0)
    file_lines_cat_1 = catalog_1.readlines()
    words_in_catalog_1 = get_words_catalog(file_lines_cat_1)

    cat_2_text = catalog_2.read()
    catalog_2.seek(0)
    file_lines_cat_2 = catalog_2.readlines()
    words_in_catalog_2 = get_words_catalog(file_lines_cat_2)

    total_words = words_in_catalog_1 + words_in_catalog_2

    common_word_list = set(words_in_catalog_2) & set(words_in_catalog_1)
    words_unique_to_cat_1 = set(total_words) - set(common_word_list) - set(words_in_catalog_2)
    words_unique_to_cat_2 = set(total_words) - set(common_word_list) - set(words_in_catalog_1)
    total_count_tqdm = len(
        [i for i in itertools.zip_longest(common_word_list, words_unique_to_cat_1, words_unique_to_cat_2)])
    start = time.time()

    for common_word, unique_word_cat1, unique_word_cat2 in tqdm(
            itertools.zip_longest(common_word_list, words_unique_to_cat_1, words_unique_to_cat_2),
            total=total_count_tqdm):

        if common_word is not None and common_word is not '\n':
            metadata_word_cat2 = get_param(common_word, cat_2_text, inverted_doc_2)
            metadata_word_cat1 = get_param(common_word, cat_1_text, inverted_doc_1)
            combined_meta_data = '{},{}'.format(metadata_word_cat1, metadata_word_cat2)
            cursor = write_inverted_doc(common_word, cursor, combined_meta_data, merged_inverted_doc, merged_catalog)

        if unique_word_cat1 is not None and unique_word_cat1 is not '\n':
            metadata_word_cat1 = get_param(unique_word_cat1, cat_1_text, inverted_doc_1)
            cursor = write_inverted_doc(unique_word_cat1, cursor, metadata_word_cat1, merged_inverted_doc,
                                        merged_catalog)

        if unique_word_cat2 is not None and unique_word_cat2 is not '\n':
            metadata_word_cat2 = get_param(unique_word_cat2, cat_2_text, inverted_doc_2)
            cursor = write_inverted_doc(unique_word_cat2, cursor, metadata_word_cat2, merged_inverted_doc,
                                        merged_catalog)
    duration = time.time() - start
    print(duration)


def merge():
    cat_files = sorted(glob.glob(str(PATH_TO_CATALOG / '*')))
    inverted_files = sorted(glob.glob(str(PATH_TO_INVERTED / '*')))
    cursor = 0
    merge_level = 1
    while len(cat_files) > 1:
        cat_file_remaining = None
        inverted_file_remaining = None
        merge_file_count = 1
        for ind in range(0, len(cat_files), 2):
            if ind >= len(cat_files) - 1:
                cat_file_remaining = cat_files[ind]
                inverted_file_remaining = inverted_files[ind]
                continue
            merged_catalog = open(
                PATH_TO_MERGED_CATALOG / 'merged_catalog_L{}_F{}'.format(merge_level, merge_file_count), 'a')
            merged_catalog.write('\n')
            merged_inverted_doc = open(
                PATH_TO_MERGED_INVERTED / 'inverted_doc_L{}_F{}'.format(merge_level, merge_file_count), 'a')
            cat_1 = open(cat_files[ind])
            cat_2 = open(cat_files[ind + 1])
            inverted_1 = open(inverted_files[ind])
            inverted_2 = open(inverted_files[ind + 1])
            print ('Using Catalogs \n {}\n {}'.format(cat_files[ind], cat_files[ind+1]))
            print('Using Inverted doc \n {}\n {}'.format(inverted_files[ind], inverted_files[ind + 1]))
            merge_doc(cat_1, cat_2, merged_catalog, inverted_1, inverted_2, merged_inverted_doc, cursor)
            merge_file_count += 1
            merged_catalog.close()
            merged_inverted_doc.close()
        if inverted_file_remaining and cat_file_remaining:
            shutil.copy(cat_file_remaining,
                        PATH_TO_MERGED_CATALOG / 'merged_catalog_L{}_F{}'.format(merge_level, merge_file_count))

            shutil.copy(inverted_file_remaining,
                        PATH_TO_MERGED_INVERTED / 'inverted_doc_L{}_F{}'.format(merge_level, merge_file_count))
        cat_files = sorted(glob.glob(str(PATH_TO_MERGED_CATALOG / '*_L{}*'.format(merge_level))))
        inverted_files = sorted(glob.glob(str(PATH_TO_MERGED_INVERTED / '*L{}*'.format(merge_level))))
        print('level {} of merging done'.format(merge_level))
        merge_level += 1


def get_vocab_size():
    catalogs = glob.glob(str(PATH_TO_CATALOG / '*'))
    vocab = []
    for file in catalogs:
        if '7z' in file:
            continue
        f = open(file, 'r')
        vocab += get_words_catalog(f.readlines())
    return len(set(vocab))


def read_data(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def compress_docs():
    cursor = 0
    try:
        inverted_file = open(PATH_TO_MERGED_INVERTED / 'inverted_doc_L7_F1')
        catalog_file = open(PATH_TO_MERGED_CATALOG / 'merged_catalog_L7_F1')

        inverted_file_compressed = open(PATH_TO_MERGED_INVERTED_COMPRESSED / 'compressed_inverted_doc_L7_F1', 'a')
        catalog_file_compressed = open(PATH_TO_MERGED_CATALOG_COMPRESSED / 'compressed_merged_catalog_L7_F1', 'a')
        catalog_file_compressed.write('\n')

        catalog_read = catalog_file.read()
        catalog_file.seek(0)
        for line in tqdm(catalog_file.readlines()):
            if line == '\n':
                continue
            word = line.split(':')[0]
            param = get_param(word, catalog_read, inverted_file)
            cursor = write_inverted_doc(word, cursor, param, inverted_file_compressed, catalog_file_compressed, True)

    except Exception :
        print("Merged catalog or inverted file not found. Please delete all catalogs and inverted docs before "
              "proceeding")
    pass


def decompress_docs():
    cursor = 0
    inverted_file_compressed = open(PATH_TO_MERGED_INVERTED / 'compressed_inverted_doc_L7_F1')
    catalog_file_compressed = open(PATH_TO_MERGED_CATALOG / 'compressed_merged_catalog_L7_F1')
    catalog_file = open(PATH_TO_MERGED_CATALOG / 'merged_catalog_L7_F1')

    inverted_file_decompressed = open(PATH_TO_MERGED_INVERTED_COMPRESSED / 'decompressed_inverted_doc_L7_F1', 'a')
    catalog_file_decompressed = open(PATH_TO_MERGED_CATALOG_COMPRESSED / 'decompressed_merged_catalog_L7_F1', 'a')
    for line in tqdm(catalog_file.readlines()):
        if line == '\n':
            continue
        word = line.split(':')[0]
        param = get_param(word, catalog_file_compressed.read(), inverted_file_compressed, compress=True)
        cursor = write_inverted_doc(word, cursor, param, inverted_file_decompressed, catalog_file_decompressed, False)

if __name__ == "__main__":
    doc_map = read_data(PATH_TO_DOC_INDEX)
    merge()
    cur = 0


