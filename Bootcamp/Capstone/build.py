import subprocess

def main():
	for cmd in ['pip3 install paramiko --user', 'pip3 install pandas --user', 'pip3 install numpy --user', 'pip3 install scipy --user', 'pip3 install gensim --user', 'pip3 install boto3 --user', 'pip3 install matplotlib --user', 'pip3 install sklearn --user', 'pip3 install seaborn --user']:
		subprocess.call(cmd, shell = True)

	cmd = 'aws s3 sync s3://agu-thinkful-capstone/models models'
	subprocess.call(cmd, shell = True)
	print('copied models')

	# cmd = 'aws s3 sync s3://agu-thinkful-capstone/closest_sections closest_sections'
	# subprocess.call(cmd, shell = True)
	# print('copied closest_sections')

	cmd = 'aws s3 cp s3://agu-thinkful-capstone/agu_df.csv agu_df.csv'
	subprocess.call(cmd, shell = True)
	print('copied agu_df')

	cmd = 'sudo mkdir clustering'
	subprocess.call(cmd, shell = True)

	# cmd = 'sudo mkdir sims_summary'
	# subprocess.call(cmd, shell = True)
	# cmd = 'sudo chmod -R 757 sims_summary'
	# subprocess.call(cmd, shell = True)
	# for ik in range(1, 24): 
	# 	cmd = 'sudo mkdir sims_summary/sims_'+str(ik)
	# 	subprocess.call(cmd, shell = True)

	# for feature in ['title_features', 'abstract_features']:
	# 	cmd = 'sudo mkdir sims_'+feature
	# 	subprocess.call(cmd, shell = True)
	# 	cmd = 'sudo chmod -R 757 sims_'+ feature
	# 	subprocess.call(cmd, shell = True)

	# 	for ik in range(1, 24): 
	# 		cmd = ' '.join(['sudo mkdir sims_', feature, '/sims_', str(ik)])
	# 		subprocess.call(cmd, shell = True)
	

if __name__ == '__main__':
   main()