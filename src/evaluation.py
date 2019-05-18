import csv
import numpy as np
import pandas as pd
from metrics import *

def evaluate(predict_path, answer_path, average=False, is_target_with_header=False, is_ranked_list_with_header=False, ranked_list_cols=None):
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

    # drop [user_id, event_time]
    if is_target_with_header:
        targets = df_ans.loc[1:,1] # drop header and choose the column of targets
    else:
        targets = df_ans.loc[:,1] # choose the column of targets
    
    if is_ranked_list_with_header:
        ranked_lists = df_pre.loc[1,:] # drop header
    else:
        ranked_lists = df_pre.loc[:,:] 

    if ranked_list_cols:
        ranked_lists = ranked_lists.loc[:,ranked_list_cols]  # choose columns of ranked items

    targets = targets.values.astype(int) # convert dataFrame to list
    ranked_lists = ranked_lists.values.astype(int)
    
    # get scores of list/average
    hr = get_hit_ratio(ranked_lists,targets,average)
    mrr = get_MRR(ranked_lists,targets,average)
    ndcg = get_NDCG(ranked_lists,targets,average)
    return {'hr':hr,'mrr':mrr,'ndcg':ndcg}