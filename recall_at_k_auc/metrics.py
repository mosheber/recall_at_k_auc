import numpy as np
from sklearn.metrics import auc

def precision_and_recall_at_k(y_true,y_score):
  """
  Calculates the recall@k and precision@k, defined as:

  recall@k = (# of relevant items in first k places) / (# of relevant elemetns overall)
  precision@k = (# of relevant items in first k places) / (k + 1) ; k starts at index 0

  input:
  y_true : list of the true labels of each item, binary only (0 or 1).
  y_score : list of the scores given to each item.

  lists/arrays must be of the same length!

  output:

  precision_at_ks : the precision@k scores.
  recall_at_ks : the recall@k scores.
  ks : list of all the ks from 0 to len(y_true).
  total_positive_count : number of positive values in y_true.
  """
  column_matrix = np.column_stack([y_true,y_score])  

  #sort in descending order
  ind=np.argsort(column_matrix[:,-1])
  column_matrix_sorted=column_matrix[ind][::-1]

  total_positive_count = sum(y_true)
  
  ks = list(range(int(column_matrix_sorted.shape[0])))

  positive_count_at_k = 0
  recall_at_ks = []
  precision_at_ks = []
  for k in ks:
    if(column_matrix_sorted[k][0]==1):
      positive_count_at_k += 1
      
    current_recall_at_k = positive_count_at_k/total_positive_count
    current_precision_at_k = positive_count_at_k/(k+1)

    recall_at_ks.append(current_recall_at_k)
    precision_at_ks.append(current_precision_at_k)
  return precision_at_ks, recall_at_ks, ks, total_positive_count

def calc_perfect_auc_at_k(k,when_reaches_max,height):
  """
  Calculates the AUC of the best recall@k AUC possible.

  input: 

  k : the k value until which to calculate the AUC.
  when_reaches_max : the value of k in which the perfect recall@k graph should reach its max value (usually 1).
  height : the max value (usually 1) of the recall@k graph.

  output:

  the AUC of the best recall@k AUC possible
  """
  if(k<=when_reaches_max):
    # calc the area of the triagle
    return ((when_reaches_max*height)/2)*(k/when_reaches_max)**2
  # calc the area of the full triagle plus the rectangle
  return ((when_reaches_max*height)/2) + (k-when_reaches_max)*height


def get_recall_at_k_auc(ks,recall_at_ks,total_positive_count,k=None):
  """
  Calculates the recall@k AUC, normalized by the best recall@k AUC possible, defined as:

  recall@k AUC : AUC of the recall@k graph 
  perfect recall@k AUC : the area of the trapezoid created for the best recall@k graph.
                         which gets: recall@total_number_of_1_in_y_true = 1.0
  If k is not None, the function calculates the AUC of both the actual and perfect recall@k 
  only until k.

  input:
  recall_at_ks : the recall@k scores.
  ks : list of all the ks from 0 to len(y_true).
  total_positive_count : number of positive values in y_true.
  k : the k until which to calculate the AUC. If k is not None, the function calculates 
      the AUC of both the actual and perfect recall@k only until k.

  output:

  metric : the recall@k AUC, normalized by the best recall@k AUC possible.
  """
  k = len(recall_at_ks) if k is None else k

  ks = ks if k is None else ks[:k]
  recall_at_ks = recall_at_ks if k is None else recall_at_ks[:k]

  height = 1.0 # since recall@k reaches max at 1
  width = len(ks)

  recall_at_auc = auc(ks,recall_at_ks)

  perfect_area =  calc_perfect_auc_at_k(width,total_positive_count,height)

  metric = recall_at_auc / perfect_area 

  return metric 