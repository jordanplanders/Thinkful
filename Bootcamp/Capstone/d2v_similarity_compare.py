import pandas as pd
from scipy import stats
pathout = 'sims_summary/resorted2/'

# Goes through a set (size>2) and identifies the best performing model-feature-section comb (or set of combs)
def find_best(sets):
    
    # takes model-feature-section combination that is untested, and tests it against the prior winners
    # returns one winner if there is a statistically significant one, or set of winners if there are 
    # more than one that are not statistically different and score best
    def tmp(limbo, winners, new_set_file):
        if len(winners)>0:
            set_list = winners
        else:
            set_list = limbo
        
        limbo = [new_set_file]
        winners = []
        new_participant = pd.read_csv(pathout+new_set_file)
        for candidate_file in set_list:
            candidate = pd.read_csv(pathout+candidate_file)
            if stats.ttest_rel(candidate['mean'], new_participant['mean'])[1]<.05:
                if candidate['mean'].mean() > new_participant['mean'].mean():
                    winners.append(candidate_file)
                else:
                    winners.append(new_set_file)
            else:
                limbo.append(candidate_file)
        return limbo, winners
    
    
    limbo = [sets[0]]
    winners = []
    
    for new_set_file in sets[1:]:
        limbo, winners = tmp(limbo, winners, new_set_file )
    
    if len(winners)>0:
        return winners
    else:
        return limbo



# pull sets for a particular cell, calculate winner(s) of cell
def get_cell_best(crit1, crit2, d_cell_best, df_1):
    for colpair in [[crit1, crit2], [crit2, crit1]]:
        col1 = colpair[0]
        col2 = colpair[1]
        for key_1b in d_cell_best[col1].keys():
            for key2 in d_cell_best[col1][key_1b].keys():
                files = find_best(df_1[(df_1[col1] == key_1b) & (df_1[col2] == key2)]['filename'].tolist()[0])
                d_cell_best[col1][key_1b][key2] = list(set(files))
    return d_cell_best



# pull winners from cells in standard table in either a row or column, calculate row or column winnner(s)
def get_each_row_col_best(crit1, crit2, d_cell_best):
    d_each_row_each_col_best = {}
    for col in [crit1, crit2]:
        d_each_row_each_col_best[col]= {}
        for key in d_cell_best[col].keys():
            files = []
            for key2 in d_cell_best[col][key]:
                files += d_cell_best[col][key][key2]
            d_each_row_each_col_best[col][key] = list(set(find_best(list(set(files)))))#d_each_row_each_col_best[col][key])
    return d_each_row_each_col_best



# pull winners from summary col or row to calculate total row-wise or col-wise winner
def get_rows_cols_best(crit1, crit2, d_each_row_each_col_best):
    d_rows_cols_best = {}
    for col in [crit1, crit2]:
        col_set = []
        for key in d_each_row_each_col_best[col].keys():
            col_set += d_each_row_each_col_best[col][key] 
        d_rows_cols_best[col] = list(set(find_best(col_set)))
    return d_rows_cols_best



# pull winners of summary row or col and record best
def get_all_best(crit1, crit2, d_rows_cols_best):
    return list(set(find_best([d_rows_cols_best[crit2]+d_rows_cols_best[crit1]])[0]))



# makes a dataframe for visualizing summary information
def make_file_df(crit1, crit2, file_df):
    d = {crit:[] for crit in [crit1, crit2, 'filename']}

    crit1_grps = file_df.groupby(crit1)
    for grp_1 in crit1_grps:
        crit12_grps = grp_1[1].groupby(crit2)
        for grp_12 in crit12_grps:
            d[crit1].append(grp_1[0])
            d[crit2].append(grp_12[0])
            d['filename'].append(grp_12[1]['filename'].tolist())

    df_1 = pd.DataFrame(d)
    return df_1



# Runs all helper functions to calculate summary table
def make_df_summary(crit1, crit2, d_cell_best, file_df):
    df_1 = make_file_df(crit1, crit2, file_df)
    d_cell_best = get_cell_best(crit1, crit2, d_cell_best, df_1)

    d_each_row_each_col_best = get_each_row_col_best(crit1, crit2, d_cell_best)

    d_rows_cols_best = get_rows_cols_best(crit1, crit2, d_each_row_each_col_best)

    best = get_all_best(crit1, crit2, d_rows_cols_best)

    # fix formatting:
    crit2_features = d_cell_best[crit2].keys()
    crit1_features = d_cell_best[crit1].keys()
    for crit in [[crit1, crit1_features], [crit2, crit2_features]]:
        d_rows_cols_best[crit[0]] = [d_rows_cols_best[crit[0]]]
        for feature in crit[1]:
            d_each_row_each_col_best[crit[0]][feature] = [d_each_row_each_col_best[crit[0]][feature]]
            for key in d_cell_best[crit[0]][feature].keys():
                d_cell_best[crit[0]][feature][key] = [d_cell_best[crit[0]][feature][key]]
    
    # prepare table
    summary_df = pd.concat([pd.DataFrame(d_cell_best[crit2][feature]) for feature in crit2_features]+ [pd.DataFrame(d_each_row_each_col_best[crit1])])
    summary_df['row summary'] = [d_each_row_each_col_best[crit2][feature][0] for feature in crit2_features] + [list(set(d_rows_cols_best[crit2][0]+d_rows_cols_best[crit1][0]))]
    summary_df.insert(0, 'labels', [feature for feature in crit2_features]+['col summary'])
    
    # clean up labels
    for col in [x for x in summary_df.columns if x != 'labels']:
        new_labels = []
        for ik in range(len(summary_df[col])):
            elements = summary_df[col].iloc[ik]
            new_cell_labels = []
            for element in elements:
                element = element.replace('sim_summary_', '').replace('.csv', '')
                if str(col) in element:
                    element = element.replace(str(col), '')
                if col != 'row summary':
                    if summary_df['labels'].iloc[ik] in element:
                        element = element.replace(summary_df['labels'].iloc[ik], '')
                element = element.strip('_')
                element = element.replace('title_features', 'TF').replace('abstract_features', 'AF')
                new_cell_labels.append(element)
            new_labels.append(new_cell_labels)
        summary_df[col] = new_labels

    print('Best:', best)
    return summary_df, best

