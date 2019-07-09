import sys
import os
import subprocess


def main():
	start_ind = 0
	end_ind = 1
		# for feature in ['abstract_features']: #['title_features', 'abstract_features']:

	cmd = ' '.join(['echo ',' '.join([str(ip) for ip in range(start_ind,end_ind, 1)]), ' | xargs -n 1 -P 1 python3 cluster_analysis2.py '])
		# 	subprocess.call(cmd, shell = True)
		# 	print('finished similarities for '+feature+str(ik))
    	
			# cmd = ' '.join(['aws s3 sync sims_'+feature+'/sims_'+str(ik), 's3://agu-thinkful-capstone/sims_'+feature+'/sims_'+str(ik)])
			# subprocess.call(cmd, shell = True)
			# print('copied similarities for '+feature)


		# cmd = 'aws s3 sync s3://agu-thinkful-capstone/sims_title_features/sims_'+str(ik)+' ~/home/ec2-user/sims_title_features/sims_'+str(ik)
		# subprocess.call(cmd, shell = True)
		# cmd = 'aws s3 sync s3://agu-thinkful-capstone/sims_abstract_features/sims_'+str(ik)+' ~/home/ec2-user/sims_abstract_features/sims_'+str(ik)
		# subprocess.call(cmd, shell = True)
		# cmd = ' '.join(['echo ',' '.join([str(ip) for ip in range(start_ind,end_ind, 1000)]), ' | xargs -n 1 -P 4 python3 sims_exp.py ', str(ik)]) 
	subprocess.call(cmd, shell = True)
	print('ran and copied cluster and sl calculations')

if __name__ == '__main__':
   main()