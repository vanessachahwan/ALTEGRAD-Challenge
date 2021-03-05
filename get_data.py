import os
import shutil
import argparse

parser = argparse.ArgumentParser(description='Download challenge data')
parser.add_argument('--precomputed', default=1,
                    help='Whether we download the precomputed data or we download the challenge data and we compute the necessary data for the model')

args = parser.parse_args()
precomputed = int(args.precomputed)

dir = 'data'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)
os.chdir(dir)
if precomputed == 1:
	os.system('kaggle datasets download vanessachahwan/altegrad-challenge-2020-bmv-data')
	os.system('tar -xf altegrad-challenge-2020-bmv-data.zip')
	os.remove('altegrad-challenge-2020-bmv-data.zip')
else:
	os.system('kaggle datasets download vanessachahwan/altegrad-challenge-2020-bmv-data -f abstracts.txt')
	os.system('tar -xf abstracts.txt.zip')
	os.remove('abstracts.txt.zip')
	os.system('kaggle datasets download vanessachahwan/altegrad-challenge-2020-bmv-data -f author_papers.txt')
	os.system('tar -xf author_papers.txt.zip')
	os.remove('author_papers.txt.zip')
	os.system(
	    'kaggle datasets download vanessachahwan/altegrad-challenge-2020-bmv-data -f train.csv')
	os.system(
	    'kaggle datasets download vanessachahwan/altegrad-challenge-2020-bmv-data -f test.csv')
	os.system('tar -xf test.csv.zip')
	os.remove('test.csv.zip')
	os.chdir('../code')
	os.system('python paper_representations.py dm')
	os.system('python author_representations.py dm')
	os.system('python paper_representations.py dbow')
	os.system('python author_representations.py dbow')
	os.system('python create_graphs.py')
	os.system('python papers_features.py')
	os.system('python graph_features.py')
	os.system('python node2vec.py')
	os.system('python author_node2vec.py pg')
	os.system('python author_node2vec.py wpg')
	os.system('python preprocessing_node2vec.py')
