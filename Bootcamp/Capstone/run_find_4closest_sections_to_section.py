import sys
import os
import subprocess


def main():
	# for ik in range(8, 14):
		for feature in ['title_features']:

			cmd = ' '.join(['echo ',' '.join([str(ip) for ip in range(30, 38, 1)]), ' | xargs -n 1 -P 4 python3 find_4closest_sections_to_section.py ', feature])
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

if __name__ == '__main__':
   main()

