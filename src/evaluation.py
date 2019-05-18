import csv
import numpy as np
import pandas as pd
from metrics import *


def evaluate(predict_path, answer_path, sort_by_col=None, average=False, is_target_with_header=False, is_ranked_list_with_header=False, ranked_list_cols=None, debug=False):
    """
    Parameters:
    ----------
    predict_path: path for predictions in csv format
    answer_path:  path for answer in csv format
    Format:
        prediction: n*(2+k) 2d-array
            n: number of test instances
            k: number of items recommended
            user_id, event_time, rec_1, rec_2, ... , rec_k
        answer: n*k 2d-array
            n: number of test instances
            k: columns = [user_id, ans_id,..., event_time]
    Return:
    ----------
    Dictionary: 
        type(key): str,
        type(value): float/[float]
        if average == True:
            {'hr':score,'mrr':score,'ndcg':score}
        else #average == False:
            {'hr':[score...],'mrr':[score...],'ndcg':[score...]}
    """
    
    df_pre = pd.read_csv(predict_path, header=None)
    df_ans = pd.read_csv(answer_path, header=None)
    if debug:
        print("answer info:",df_ans.shape,'\n')
        print(df_ans.head(3))
        print("\n------\n")
        print("prediction info:",df_pre.shape,'\n')
        print(df_pre.head(3))

    # drop [user_id, event_time]
    if is_target_with_header:
        targets = df_ans.loc[1:,:] # drop header
    else:
        targets = df_ans.loc[:,:] # retain header
    
    if is_ranked_list_with_header:
        ranked_lists = df_pre.loc[1,:] # drop header
    else:
        ranked_lists = df_pre.loc[:,:] # drop header

    if sort_by_col == None:
        targets = targets.loc[:,1]
    else:
        targets.sort_values(by=sort_by_col,inplace=True)
        ranked_lists.sort_values(by=sort_by_col,inplace=True)
        if debug:
            print("\n====================================================\n")
            print("check sorting result")
            print("target (first, last):",targets.head(1)[sort_by_col].values[0],targets.tail(1)[sort_by_col].values[0])
            print("predict (first, last):",ranked_lists.head(1)[sort_by_col].values[0],ranked_lists.tail(1)[sort_by_col].values[0])
        targets = targets.loc[:,1]
        
    if ranked_list_cols:
        ranked_lists = ranked_lists.loc[:,ranked_list_cols]  # choose columns of ranked items

    targets = targets.values.astype(int) # convert dataFrame to list
    ranked_lists = ranked_lists.values.astype(int)
    if debug:
        print("\n====================================================\n")
        print("targets info:",targets.shape,targets.dtype,'\n')
        print(targets[:3])
        print("\n------\n")
        print("\nranked_lists info:",ranked_lists.shape,ranked_lists.dtype,'\n')
        print(ranked_lists[:3])

    # get scores of list/average
    hr = get_hit_ratio(ranked_lists,targets,average)
    mrr = get_MRR(ranked_lists,targets,average)
    ndcg = get_NDCG(ranked_lists,targets,average)
    return {'hr':hr,'mrr':mrr,'ndcg':ndcg}