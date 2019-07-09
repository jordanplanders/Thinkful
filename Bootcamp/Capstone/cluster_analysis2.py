from gensim.models import Doc2Vec
import sys, os, time, subprocess

from sklearn.cluster import KMeans, AgglomerativeClustering#, SpectralClustering, DBSCAN
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc, silhouette_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

import pandas as pd
import numpy as np
import csv
# model feature clustering_key num_clusters function time

def main():
	def move_file_to_s3(file_path):
		cmd = ' '.join(['aws s3 cp ', file_path, 's3://agu-thinkful-capstone/'+file_path])
		subprocess.call(cmd, shell = True)
		print('copied '+file_path+' to s3', file=sys.stderr)

		cmd = 'rm '+file_path
		print(cmd, file= sys.stderr)
		subprocess.call(cmd, shell = True)
		print('removed '+file_path+' from ec2', file=sys.stderr)

    
	def make_clusters(clustering_key,param_range, doc_vecs_df, meta_data, csv_file_name, **kwargs):
		nclusts = []
		param = []
		sscores = [] 
		avg_obs = []
		obs_std = []
		models = []
		labels = []


		for no in param_range:
			t1 = time.time()
			if clustering_key =='kmeans':
				model = KMeans(n_clusters=no, random_state=43).fit(doc_vecs_df)
#             elif clustering_key == 'DBSCAN':
#                 model = DBSCAN(eps=no, algorithm='auto').fit(doc_vecs_df)
#             elif clustering_key == 'DBSCAN_kd':
#                 model = DBSCAN(eps=no, algorithm='kd_tree').fit(doc_vecs_df)
			elif clustering_key == 'agglomerative':
				model = AgglomerativeClustering(linkage = 'ward', n_clusters= no).fit(doc_vecs_df)
                #AffinityPropagation(affinity='euclidean', convergence_iter=15, copy=True, damping=no, max_iter=200, preference=None, verbose=False).fit(doc_vecs_df)
			fdv_clusters = model.labels_#predict(doc_vecs_df)
			# print('label set', set(fdv_clusters))
			nclusts.append(len(list(set(fdv_clusters))))
			param.append(no)
			sscores.append(silhouette_score(doc_vecs_df, fdv_clusters, metric='cosine'))
			avg_obs.append(pd.value_counts(fdv_clusters).mean())
			obs_std.append(pd.value_counts(fdv_clusters).std())
			models.append(model)
			labels.append(fdv_clusters)
			# if no == param_range[0]:
			# 	with open(csv_file_name, 'w') as csv_file:
			# 		writer = csv.writer(csv_file, delimiter=',')
			# 		writer.writerow(meta_data+ [no, 'make_clusters', time.time()-t1])
			# else:
			with open(csv_file_name, 'a') as csv_file:
				writer = csv.writer(csv_file, delimiter=',')
				writer.writerow(meta_data+ [no, 'make_clusters', time.time()-t1])
			print(meta_data+[no, 'make_clusters', time.time()-t1], file=sys.stderr)

		return models, pd.DataFrame({'nclusts':nclusts, 'param':param, 'sscores':sscores, 'avg_obs':avg_obs, 'obs_std':obs_std, 'labels':labels})


	def make_sil_plot(df, **kwargs):
		obsstd1 = [ i+j for i,j in zip(df['avg_obs'],df['obs_std'])]
		stdneg = [ i-j for i,j in zip(df['avg_obs'],df['obs_std'])]

		fig, ax = fig, ax = plt.subplots(figsize=(6,4))

		ax2 = ax.twinx()
		ax = sns.scatterplot(x = 'param', y = 'sscores', data = df,hue='nclusts', label='Sil. Score', ax=ax)

		ax2 =sns.lineplot(df['param'], df['avg_obs'],color='purple',label='Avg obs per cluster', linewidth=2)
		sns.lineplot(df['param'],obsstd1,color='r',label='+/- Std Avg Obs', linewidth=.6)
		sns.lineplot(df['param'],stdneg,color='r', linewidth=.6)

		ax.set_ylabel('Sil. Score')
		ax.set_xlabel('Number of Clusters')
		ax.axvline(x=df['param'].iloc[df['sscores'].idxmax()],color='r',linestyle='dotted')

		ax.legend(loc='lower left')
		ax2.legend(loc='upper right')
		plt.ylabel('Average Observation per cluster')
		plt.xlabel('Number of Clusters')
		plt.title('Silhouette Scores by Number of Clusters',fontsize=20)

		if 'save_name' in kwargs:
			plt.savefig(kwargs['save_name'], format='svg', bbox_inches='tight')
    
	def make_train_test_inds(df_for_inds):
		balanced_inds_train = []
		for grp in df_for_inds.groupby('labels'):
			balanced_inds_train += grp[1].sample(frac = .8).index.tolist()

		balanced_inds_test = df_for_inds[~df_for_inds.index.isin(balanced_inds_train)].index.tolist()
		return balanced_inds_train, balanced_inds_test

	def make_d2v_df(model, df):
		return pd.DataFrame([model.docvecs[ik] for ik in range(len(df))])

	params_df = pd.read_csv('feature_modelnum_clustermodel2.csv')
	comb_num = int(sys.argv[1])
	clustering_key = params_df.iloc[comb_num]['clustering_key']
	model_num = str(params_df.iloc[comb_num]['model_num'])
	feature = params_df.iloc[comb_num]['feature']
	csv_file_name = 'clustering/ml_logging_d2v_'+clustering_key+'_'+model_num+ '_'+feature+'.csv'
	cmd = 'touch '+ csv_file_name
	subprocess.call(cmd, shell = True)
	meta_data = [model_num, feature, clustering_key]

    # Returns doc2vec modelled vectors
	df = pd.read_csv('agu_df.csv')
	model = Doc2Vec.load('models/d2v_'+feature+'_params_'+model_num+'.model')
	doc_vecs_df = make_d2v_df(model, df)

	param_d = {'kmeans': list(range(2, 27, 1)), 'agglomerative': list(range(2, 27, 1))}#, 'DBSCAN': np.arange(10, 10.78, .03).tolist(), 'DBSCAN_kd': np.arange(10, 10.78, .03).tolist()}
	filter_cols =  ['labels', 'section', 'session', 'labels_sub', 'identifier']

    # kmeans_mdls, df_clusters = make_clusters(doc_vecs_df.iloc[:, ~doc_vecs_df.columns.isin(filter_cols)])
	kmeans_mdls, df_clusters = make_clusters(clustering_key, param_d[clustering_key], doc_vecs_df.iloc[:, ~doc_vecs_df.columns.isin(filter_cols)],meta_data, csv_file_name)
	clustering_csv_file = 'clustering/clustering_d2v_'+clustering_key+'_'+model_num+ '_'+feature+'.csv'
	df_clusters.to_csv(clustering_csv_file)
	move_file_to_s3(clustering_csv_file)

	t1 = time.time()
	silplot_file_name = 'clustering/silplot_d2v_'+clustering_key+'_'+model_num + '_'+feature+'.svg'
	make_sil_plot(df_clusters, save_name = silplot_file_name)
	move_file_to_s3(silplot_file_name)
	with open(csv_file_name, "a") as csv_file:
			writer = csv.writer(csv_file, delimiter=',')
			writer.writerow(meta_data+ [0, 'make_sil_plot', time.time()-t1])
    

	d = {'clust_param':[], 'X_train_inds':[], 'X_test_inds':[], 'y_test_cl':[], 'y_test_sl':[], 
		'svc_y_pred_cl':[], 'svc_test_score_cl': [], 'svc_train_score_cl': [],'gbc_y_pred_cl':[], 'gbc_test_score_cl':[], 'gbc_train_score_cl':[],
		'svc_y_pred_sl':[], 'svc_test_score_sl': [], 'svc_train_score_sl': [],'gbc_y_pred_sl':[], 'gbc_test_score_sl':[], 'gbc_train_score_sl':[] }
    
	doc_vecs_df['section'] = df['section']
	for ik in range(len(df_clusters)):
		doc_vecs_df['labels'] = df_clusters.iloc[ik]['labels']
		X_train_inds, X_test_inds = make_train_test_inds(doc_vecs_df)
#         supclass_X_train, supclass_X_test =  doc_vecs_df[inds_d['X_train_inds']], doc_vecs_df[inds_d['X_test_inds']]
		supclass_X_train, supclass_X_test =  doc_vecs_df.iloc[X_train_inds], doc_vecs_df.iloc[X_test_inds]

		supclass_y_train_cl, supclass_y_test_cl = supclass_X_train['labels'], supclass_X_test['labels']
		supclass_y_train_sl, supclass_y_test_sl = supclass_X_train['section'], supclass_X_test['section']

		supclass_X_train = supclass_X_train.iloc[:, ~supclass_X_train.columns.isin(filter_cols)]
		supclass_X_test = supclass_X_test.iloc[:, ~supclass_X_test.columns.isin(filter_cols)]

		SC_model_d ={'svc': svm.SVC(), 'gbc': GradientBoostingClassifier()}
		for key in SC_model_d.keys():
			for label_set in [['cl' , supclass_y_train_cl, supclass_y_test_cl], ['sl' , supclass_y_train_sl, supclass_y_test_sl]]:
				t1 = time.time()
				SC_model_d[key].fit(supclass_X_train, label_set[1])
				y_pred = SC_model_d[key].predict(supclass_X_test)
				# print(df_clusters.iloc[ik]['nclusts'], key, label_set[0],'Doc2Vec', 'Training set score:',SC_model_d[key].score(supclass_X_train, label_set[1]), 'Test set score:', SC_model_d[key].score(supclass_X_test, label_set[2]) , file=sys.stderr)
				d[key+'_train_score_'+label_set[0]].append(SC_model_d[key].score(supclass_X_train, label_set[1]))
				d[key+'_test_score_'+label_set[0]].append(SC_model_d[key].score(supclass_X_test, label_set[2]))
				d[key+'_y_pred_'+label_set[0]].append([y_pred])
				with open(csv_file_name, "a") as csv_file:
					writer = csv.writer(csv_file, delimiter=',')
					writer.writerow(meta_data+ [df_clusters.iloc[ik]['nclusts'], key+'_'+label_set[0], time.time()-t1])
		d['X_train_inds'].append(X_train_inds)
		d['X_test_inds'].append(X_test_inds)
		d['clust_param'].append(df_clusters.iloc[ik]['nclusts'])
		d['y_test_sl'].append(supclass_y_test_sl)
		d['y_test_cl'].append(supclass_y_test_cl)

	classifier_csv_file = 'clustering/classifiers_d2v_'+clustering_key+'_'+model_num + '_'+feature+'.csv'
	pd.DataFrame(d).to_csv(classifier_csv_file)
	move_file_to_s3(classifier_csv_file)
	move_file_to_s3(csv_file_name)

if __name__ == '__main__':
	main()

