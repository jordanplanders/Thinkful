
import d2v_sims
import sims_exp
import sys

def main():
	print(sys.argv[1], sys.argv[2])
	d2v_sims.main('abstract_features')
	d2v_sims.main('title_features')
	sims_exp.main()


if __name__ == '__main__':
   main()


