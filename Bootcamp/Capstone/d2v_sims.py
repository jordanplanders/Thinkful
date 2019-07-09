import sys
from gensim.models import Doc2Vec
import pandas as pd
import os
import time
import subprocess
import csv 

def main(feature):
    # check to see if this set of similarity calcs has already been done
	save_file = 'sims_summary/sims_'+sys.argv[1]+'/sims_summary_five_sections_params_'+sys.argv[1]+'_'+sys.argv[2]+'.csv'
	cmd = 'aws s3 ls '+'s3://agu-thinkful-capstone/'+save_file
	response = subprocess.call(cmd, shell = True)
	if str(response) == str(0):
		print('already calculated')	

	else:
		# if the **analysis** hasn't been done, have the similarities been calculate?
		start_ind = int(float(sys.argv[2]))
		cmd = 'aws s3 ls '+'s3://agu-thinkful-capstone/sims_'+feature+'/sims_'+sys.argv[1]+'/d2v_'+feature+'_params_'+sys.argv[1]+'_'+str(start_ind)+'.csv'
		response = subprocess.call(cmd, shell = True)
		if str(response) == str(0):
			# if the similarities have been calculated, copy them from s3 to ec2
			cmd = 'aws s3 cp '+'s3://agu-thinkful-capstone/sims_'+feature+'/sims_'+sys.argv[1]+'/d2v_'+feature+'_params_'+sys.argv[1]+'_'+str(start_ind)+'.csv' + ' sims_'+feature+'/sims_'+sys.argv[1]+'/d2v_'+feature+'_params_'+sys.argv[1]+'_'+str(start_ind)+'.csv'
			subprocess.call(cmd, shell = True)
			print('copied file over to ec2')	
		else:
            # if the similarities haven't been calculated, calculate
			print('building file from scratch')
			csv.field_size_limit(100000000)
			t1 = time.time()
			df = pd.read_csv('agu_df.csv',engine='python')
			num_abstracts = len(df)
			
			# do calcs in blocks of 1000
			df = df[start_ind:min(start_ind+1000, num_abstracts)]

			# load model
			model = Doc2Vec.load('models/d2v_'+feature+'_params_'+sys.argv[1]+'.model')
			print('type of model',type(model), file=sys.stderr)

            # do similarity calculations between each seed abstract and all other abstracts
			sims_lst = []
			for ik in range(len(df)):
				new_vector = model.infer_vector(df.iloc[ik][feature].split())
				sims = model.docvecs.most_similar([new_vector], topn=num_abstracts)
				sims_lst.append(sims)
				# print(ik)
				if ik %1000 == 0:
					print(feature, sys.argv[1], sys.argv[2], ik)

            # build dataframe and save to csv
			df_sims = pd.DataFrame({'index': df.index, 'sims':sims_lst})
			print(df_sims.shape)
					df_sims.to_csv('sims_'+feature+'/sims_'+sys.argv[1]+'/d2v_'+feature+'_params_'+sys.argv[1]+'_'+str(start_ind)+'.csv')
			print(feature, sys.argv[1], sys.argv[2] , time.time()-t1, file=sys.stderr)

			# copy to s3 bucket
			cmd = ' '.join(['aws s3 cp sims_'+feature+'/sims_'+sys.argv[1]+'/d2v_'+feature+'_params_'+sys.argv[1]+'_'+str(start_ind)+'.csv', 's3://agu-thinkful-capstone/sims_'+feature+'/sims_'+sys.argv[1]+'/d2v_'+feature+'_params_'+sys.argv[1]+'_'+str(start_ind)+'.csv'])
			subprocess.run(cmd, shell = True)



if __name__ == '__main__':
   main()

