from graph import process_sdat, process_adj_matrix

def process(dataset):

    # process_sdat(f"/mnt/ssd1/Leyuan_RA/as/project/CS4248_Group8_SarcasmDetection/data/preprocessed_stopwords_removed.csv", 
    # 'lemmatized_text_spacy',
    # '/mnt/ssd1/Leyuan_RA/as/project/CS4248_Group8_SarcasmDetection/data/featured_spacy')
    # process_sdat(f"/mnt/ssd1/Leyuan_RA/as/project/CS4248_Group8_SarcasmDetection/data/preprocessed_stopwords_removed.csv", 
    # 'lemmatized_text_nltk',
    # '/mnt/ssd1/Leyuan_RA/as/project/CS4248_Group8_SarcasmDetection/data/featured_nltk')


    process_adj_matrix(f"/mnt/ssd1/Leyuan_RA/as/project/CS4248_Group8_SarcasmDetection/data/preprocessed_stopwords_removed.csv", 
    'lemmatized_text_spacy',
    '/mnt/ssd1/Leyuan_RA/as/project/CS4248_Group8_SarcasmDetection/data/featured_spacy')
    process_adj_matrix(f"/mnt/ssd1/Leyuan_RA/as/project/CS4248_Group8_SarcasmDetection/data/preprocessed_stopwords_removed.csv", 
    'lemmatized_text_nltk',
    '/mnt/ssd1/Leyuan_RA/as/project/CS4248_Group8_SarcasmDetection/data/featured_nltk')


if __name__ == "__main__":
    process("headlines")