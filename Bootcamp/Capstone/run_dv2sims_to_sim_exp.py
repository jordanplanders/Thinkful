import sys
import os
import subprocess


def main():
	start_ind = 0
	end_ind = 25500
	for ik in range(30,34):
		# for feature in ['abstract_features']: #['title_features', 'abstract_features']:

		cmd = ' '.join(['echo ',' '.join([str(ip) for ip in range(start_ind,end_ind, 1000)]), ' | xargs -n 1 -P 4 python3 d2v_sims_regrouper.py '+str(ik)])

		subprocess.call(cmd, shell = True)
		print('ran and copied similarity calculations')

if __name__ == '__main__':
   main()

