import sys
from gensim.models import Doc2Vec
import pandas as pd
import numpy as np


def main():

    df = pd.read_csv('agu_df.csv')

    d_sections = {section: {section1: [] for section1 in set(df['section'])} for section in set(df['section'])}
    
    feature = sys.argv[1]
    model = Doc2Vec.load('models/d2v_'+feature+'_params_'+sys.argv[2]+'.model')

    for ik in range(len(df)):
        new_vector = model.infer_vector(df.iloc[ik][feature].split())
        sims = model.docvecs.most_similar([new_vector], topn=50) #N.B. This was very sensitive to class imbalance.  topn = 100 exposed the fact that certain sections just have so many more abstracts
        for pair in sims:
            d_sections[df.iloc[ik]['section']][df.iloc[int(pair[0])]['section']].append(pair[1])
        if ik %10000 == 0:
            print(ik)

    final_sections = {}
    for section in d_sections.keys():
        scores = []
        sections = []
        for section1 in d_sections[section].keys():
            if section != section1:
                sections.append(section1)
                scores.append(np.mean(d_sections[section][section1]))
        inds = np.argsort(np.array(scores))

        final_sections[section]= np.array(sections)[inds[:4]]

    pd.DataFrame(final_sections).to_csv('closest_sections/closest_sections_'+feature+'_params_'+sys.argv[2]+'.csv')
   

if __name__ == '__main__':
    main()