import sys
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk import word_tokenize
import time
import pandas as pd
import os

def main():

	def train_model(tagged_vects, params):
		max_epochs = 200
		# params['vector_size']=15
		# params['window'] = 15
		model = Doc2Vec(**params)
		model.build_vocab(tagged_vects)
		for epoch in range(max_epochs):
			model.train(tagged_vects, total_examples = model.corpus_count, epochs = model.iter)
			model.alpha -= .001
			model.min_alpha = model.alpha
		return model

	df = pd.read_csv('agu_df.csv')
	df_params = pd.read_csv('d2v_param_sets2.csv')
	params = df_params.to_dict('records')[int(float(sys.argv[2]))]
	feature = sys.argv[1]

	print(sys.argv[2],': loaded', time.time())
	tagged_X = [TaggedDocument(words = word_tokenize(comment), tags = [str(ik)]) for  ik, comment in enumerate(df[feature])]
	t1 = time.time()
	model = train_model(tagged_X, params)
	t2 = time.time()
	print(sys.argv[2],':', 'trained: {}'.format(t2-t1))

	model.save('models/d2v_'+feature+'_params_'+str(int(sys.argv[2])+30)+'.model')
	print(sys.argv[1], sys.argv[2],': saved', time.time())

if __name__ == '__main__':
	main()