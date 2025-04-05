from graph import process_sdat, process_adj_matrix
import pandas as pd
from sklearn.model_selection import train_test_split # Changed import statement to correctly import train_test_split

data=pd.read_csv('/mnt/ssd1/Leyuan_RA/as/project/CS4248_Group8_SarcasmDetection/data/features_nltk.csv')

train, test = train_test_split(data, test_size=0.2, random_state=42)
train.to_csv('/mnt/ssd1/Leyuan_RA/as/project/CS4248_Group8_SarcasmDetection/data/headline_cleaned_train.csv')
test.to_csv('/mnt/ssd1/Leyuan_RA/as/project/CS4248_Group8_SarcasmDetection/data/headline_cleaned_test.csv')

def process(dataset):
    process_sdat(f"/mnt/ssd1/Leyuan_RA/as/project/CS4248_Group8_SarcasmDetection/data/headline_cleaned_train.csv", 
    'headline_cleaned',
    '/mnt/ssd1/Leyuan_RA/as/project/CS4248_Group8_SarcasmDetection/data/headline_cleaned_train')


    process_adj_matrix(f"/mnt/ssd1/Leyuan_RA/as/project/CS4248_Group8_SarcasmDetection/data/headline_cleaned_train.csv", 
    'headline_cleaned',
    '/mnt/ssd1/Leyuan_RA/as/project/CS4248_Group8_SarcasmDetection/data/headline_cleaned_train')


    process_sdat(f"/mnt/ssd1/Leyuan_RA/as/project/CS4248_Group8_SarcasmDetection/data/headline_cleaned_test.csv", 
    'headline_cleaned',
    '/mnt/ssd1/Leyuan_RA/as/project/CS4248_Group8_SarcasmDetection/data/headline_cleaned_test')


    process_adj_matrix(f"/mnt/ssd1/Leyuan_RA/as/project/CS4248_Group8_SarcasmDetection/data/headline_cleaned_test.csv", 
    'headline_cleaned',
    '/mnt/ssd1/Leyuan_RA/as/project/CS4248_Group8_SarcasmDetection/data/headline_cleaned_test')


if __name__ == "__main__":
    process("headlines")