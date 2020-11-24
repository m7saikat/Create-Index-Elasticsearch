from pathlib import Path

INDEX = 'dataset'
DOC_TYPE = 'document'
QUERY_TYPE = 'query'
QUERY_VECTOR_FILE = "./query_term_vectors"
ROOT = Path(__file__).parents[2]

PATH_TO_CATALOG = ROOT / 'data/AP89_DATA/AP_DATA/Catalog'
PATH_TO_INVERTED = ROOT / 'data/AP89_DATA/AP_DATA/Inverted'
PATH_TO_MERGED_INVERTED = ROOT / 'data/AP89_DATA/AP_DATA/merged/Inverted'
PATH_TO_MERGED_CATALOG = ROOT / 'data/AP89_DATA/AP_DATA/merged/Catalog'

PATH_TO_MERGED_INVERTED_COMPRESSED = ROOT / 'data/AP89_DATA/AP_DATA/merged/compressed/compressed_inverted_doc_L7_F1'
PATH_TO_MERGED_CATALOG_COMPRESSED = ROOT / 'data/AP89_DATA/AP_DATA/merged/compressed/compressed_merged_catalog_L7_F1'

PATH_TO_Query = ROOT / 'data/AP89_DATA/AP_DATA/query_desc.51-100.short.txt'
PATTERN_QUERY_NO = r'^[0-9]+'
PATTERN_QUERY = r'Document.*'
PATTERN_DOCID = r'AP.*'
PATH_TO_STOPWORDS = ROOT / 'data/AP89_DATA/AP_DATA/stoplist.txt'
PATH_TO_DOCLIST = ROOT / 'data/AP89_DATA/AP_DATA/doclist.txt'
PATH_To_DATA = ROOT / 'data/AP89_DATA/AP_DATA'
PATH_TO_DOC_INDEX = PATH_To_DATA / 'docno_index'
PATH_TO_DOC_INDEX_PROXIMITY = PATH_To_DATA / 'doc_id_to_index'
NUMBER_OF_DOCUMENTS = 84678
SUM_TTF_CORPUS = 62051317
NUM_WORDS_TO_BE_SELECTED = 2
WORD_COUNT_THRESHOLD = 1
IDF_THRESHOLD = 300
ALPHA = .75

PATTERN_DOC = r'<DOC>.*?</DOC>'
PATTERN_DOC_NO = r'<DOCNO>\s(.*?)\s</DOCNO>'
PATTERN_TEXT = r'<TEXT>(.*?)</TEXT>'
PATH_TO_AP_DATASET = 'data/AP89_DATA/AP_DATA/ap89_collection'
PATH_TO_STOPWORDS_CREATE_INDEX = 'data/AP89_DATA/AP_DATA/stoplist.txt'

OUTPUT_FILE_ES = 'result_es.txt'
OUTPUT_FILE_OKAPI_TF = 'result_okapi_tf.txt'
OUTPUT_FILE_TF_IDF = 'result_tf_idf.txt'
OUTPUT_FILE_OKAPI_BM25 = 'result_okapi_bm25.txt'
OUTPUT_FILE_UNI_LAP = 'result_unigram_laplace.txt'
OUTPUT_FILE_UNI_JEL = 'result_unigram_jelinek.txt'

DOC_LEN = {}