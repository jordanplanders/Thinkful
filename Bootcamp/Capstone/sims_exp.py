import pandas as pd
import numpy as np
import csv
import itertools
import ast
import sys
import gzip
from scipy import stats
import time
import subprocess

def main():

	def get_scores_inds(row, n_abstracts, df, sections):
		scores = []
		inds = []
		ik = 0

		if len(sections)>0:
			while len(scores)<n_abstracts and ik < len(row):
				if df.iloc[int(row[ik][0])]['section'] in sections:
					scores.append(row[ik][1])
					inds.append(row[ik][0])
				ik +=1
		else:
			while len(scores)<n_abstracts and ik < len(row):
				scores.append(row[ik][1])
				inds.append(row[ik][0])
				ik +=1

		return [scores, inds]

	def get_scores_a_of_b(sim_scores_a, sim_inds_a, sim_inds_b):
		lst = [sim_scores_a[int(sim_inds_a.index(str(sim_inds_b[ip])))] for ip in range(len(sim_inds_b))]
		return lst

	save_file = 'sims_summary/sims_'+sys.argv[1]+'/sims_summary_five_sections+_params_'+sys.argv[1]+'_'+sys.argv[2]+'.csv'
	cmd = 'aws s3 ls '+'s3://agu-thinkful-capstone/'+save_file
	response = subprocess.call(cmd, shell = True)
	if str(response) == str(0):
		print('already calculated')	
	else:
		csv.field_size_limit(100000000)
		t1 = time.time()
		df = pd.read_csv('agu_df.csv', engine='python')
		df_closest_sections_tf = pd.read_csv('closest_sections/closest_sections_title_features_params_'+str(sys.argv[1])+'.csv', encoding='utf8', engine='python')
		df_closest_sections_af = pd.read_csv('closest_sections/closest_sections_abstract_features_params_'+str(sys.argv[1])+'.csv', encoding='utf8', engine='python')
		# print('df_closest_sections_tf.shape',df_closest_sections_tf.shape, file=sys.stderr)

		all_sections = {feature : {'n':[], 'ttest_p':[], 'mean':[], 'std':[]} for feature in ['title_features', 'abstract_features', 'abstracts_of_titles', 'titles_of_abstracts']}
		single_section = {feature : {'n':[], 'ttest_p':[],'mean':[], 'std':[]} for feature in ['title_features', 'abstract_features', 'abstracts_of_titles', 'titles_of_abstracts']}
		five_sections = {feature : {'n':[], 'ttest_p':[], 'mean':[], 'std':[]} for feature in ['title_features', 'abstract_features', 'abstracts_of_titles', 'titles_of_abstracts']}

		print('check point', file=sys.stderr)

		n_abstracts = 60

		with open('sims_title_features/sims_'+str(sys.argv[1])+'/d2v_title_features_params_'+str(sys.argv[1])+'_'+str(sys.argv[2])+'.csv') as f1, open('sims_abstract_features/sims_'+str(sys.argv[1])+'/d2v_abstract_features_params_'+str(sys.argv[1])+'_'+str(sys.argv[2])+'.csv') as f2:
		# with gzip.open('~/home/ec2-user/sims_title_features/sims_'+sys.argv[1]+'/d2v_title_features_params_'+sys.argv[1]+'_'+sys.argv[2]+'.csv.gz') as f1, gzip.open('~/home/ec2-user/sims_abstract_features/sims_'+sys.argv[1]+'/d2v_abstract_features_params_'+sys.argv[1]+'_'+sys.argv[2]+'.csv.gz') as f2:
			csv_titles  = csv.reader(f1)
			csv_abstracts  = csv.reader(f2)
			next(csv_titles, None)
			next(csv_abstracts, None)
			print('loaded csvs', file=sys.stderr)
			# header = csv_titles.next()
			# header = csv_features.next()

			for title_row_tmp,abstract_row_tmp in zip(csv_titles, csv_abstracts):
				# print('zipped rows', title_row_tmp, abstract_row_tmp, file=sys.stderr)
				title_row = ast.literal_eval(title_row_tmp[2])
				abstract_row = ast.literal_eval(abstract_row_tmp[2])

				title_inds = [pair[0] for pair in title_row]
				title_scores = [pair[1] for pair in title_row]

				abstract_inds = [pair[0] for pair in abstract_row]
				abstract_scores = [pair[1] for pair in abstract_row]

				ik = int(title_row_tmp[1])
				if ik%100 == 0:
					print(ik, file=sys.stderr)

				experiments = [[all_sections, {'title_features':[], 'abstract_features':[] }], 
						[single_section, {'title_features':[df.iloc[ik]['section']], 'abstract_features':[df.iloc[ik]['section']] }], 
						[five_sections, {'title_features': list(df_closest_sections_tf[df.iloc[ik]['section']]) + [df.iloc[ik]['section']], 'abstract_features':list(df_closest_sections_af[df.iloc[ik]['section']]) + [df.iloc[ik]['section']] } ]]

				# no section limitations:
				for [exp_d, sections_d] in experiments:

					inds_d = {}
					score_d = {}

					for [feature, row] in [['title_features', title_row], ['abstract_features', abstract_row]]:
						[scores, inds] = get_scores_inds(row, n_abstracts, df, sections_d[feature])
						score_d[feature] = scores
						exp_d[feature]['mean'].append(np.mean(scores))
						exp_d[feature]['std'].append(np.std(scores))
						inds_d[feature] = inds

					# abstracts of titles:
					scores = get_scores_a_of_b(abstract_scores, abstract_inds, inds_d['title_features'])
					exp_d['abstracts_of_titles']['mean'].append(np.mean(scores))
					exp_d['abstracts_of_titles']['std'].append(np.std(scores))
					exp_d['abstracts_of_titles']['ttest_p'].append(stats.ttest_rel(scores, score_d['title_features'])[1])


					# titles of abstracts:
					scores = get_scores_a_of_b(title_scores, title_inds, inds_d['abstract_features'])
					exp_d['titles_of_abstracts']['mean'].append(np.mean(scores))
					exp_d['titles_of_abstracts']['std'].append(np.std(scores))
					exp_d['titles_of_abstracts']['ttest_p'].append(stats.ttest_rel(scores, score_d['abstract_features'])[1])


			pd.DataFrame(all_sections).to_csv('sims_summary/sims_'+sys.argv[1]+'/sims_summary_all_sections_params_'+sys.argv[1]+'_'+sys.argv[2]+'.csv')
			pd.DataFrame(single_section).to_csv('sims_summary/sims_'+sys.argv[1]+'/sims_summary_single_sections_params_'+sys.argv[1]+'_'+sys.argv[2]+'.csv')
			pd.DataFrame(five_sections).to_csv('sims_summary/sims_'+sys.argv[1]+'/sims_summary_five_sections_params_'+sys.argv[1]+'_'+sys.argv[2]+'.csv')
			print(sys.argv[1], sys.argv[2], 'sims are done', time.time()-t1, file=sys.stderr)

			for exp_label in ['all_sections', 'single_sections', 'five_sections']:
				cmd = ' '.join(['aws s3 cp sims_summary/sims_'+sys.argv[1]+'/sims_summary_'+exp_label+'_params_'+sys.argv[1]+'_'+sys.argv[2]+'.csv', 's3://agu-thinkful-capstone/sims_summary/sims_'+sys.argv[1]+'/sims_summary_'+exp_label+'_params_'+sys.argv[1]+'_'+sys.argv[2]+'.csv'])
				subprocess.call(cmd, shell = True)
			print('copied similarity calculations')

		for feature in ['title_features', 'abstract_features']:
			cmd = 'rm sims_'+feature+'/sims_'+str(sys.argv[1])+'/d2v_'+feature+'_params_'+str(sys.argv[1])+'_'+str(sys.argv[2])+'.csv'
			print(cmd, file= sys.stderr)
			subprocess.call(cmd, shell = True)
		print('deleted ec2 similarity data')

if __name__ == '__main__':
   main()

	







# plot 1 needs: mean similarity of 100 title similarity for every title, mean similarity of 100 abstract similarity for every abstract
# plot 2 needs: mean and standard deviation of 100 title similarities for every title
# plot 3 needs: mean and standrad devaition of 100 abstract similarities for every abstract
# plot 4 needs: mean of 100 title similarities and mean of corresponding 100 abstract similarities
