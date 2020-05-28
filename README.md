Recall@k AUC
=======================================

Calculates the recall@k AUC, normalized by the best recall@k AUC possible.

Example Usage:
-------------

```python
import numpy as np
from recall_at_k_auc.metrics import precision_and_recall_at_k, get_recall_at_k_auc

n=1000
y_score = list(np.random.rand(n)) 
y_true = list(np.array(np.random.rand(n)>=0.5,dtype=int))

precision_at_ks, recall_at_ks, ks, total_positive_count= precision_and_recall_at_k(y_true,y_score)

metric = get_recall_at_k_auc(ks,recall_at_ks,total_positive_count,500)
```

