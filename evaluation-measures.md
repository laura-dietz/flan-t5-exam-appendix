# Implementation of Evaluation Measures
 
In the paper we are using evaluation measures using standard implementations as below.


### Rank Correlation

Using the official leaderboard as  reference, we use  Spearman's rank correlation coefficient ( `spearmanr`)  and and Kendall's Tau rank coorrelation ( `kendalltau`) from the   `scipy.stats` package.


```python

import statistics
import scipy
from scipy.stats import spearmanr, kendalltau, rankdata
import scipy.stats

def compatible_kendalltau(ranks1, ranks2)->Tuple[float,float]:
    from packaging import version

    if version.parse(scipy.__version__) >= version.parse('1.7.0'):    
    # if scipy.__version__ >= '1.7.0':
        # For scipy 1.7.0 and later
        tau, p_value = kendalltau(ranks1, ranks2)
        return tau, p_value
    else:
        # For older versions
        from scipy.stats import SignificanceResult
        result = kendalltau(ranks1, ranks2)
        return result.correlation, result.pvalue


def leaderboard_rank_correlation(systemEval:Dict[str,float], official_leaderboard:Dict[str,int])->CorrelationStats:
    methods = list(official_leaderboard.keys())
    ranks1 = [official_leaderboard[method] for method in methods]

    for method in methods:
        if not method in systemEval:
            raise RuntimeError(f'official leaderboard contains method {method}, but predicted leaderboard does not.  \nMethods in predicted leaderboard:{systemEval.keys()} \nMethods in official leaderboard {methods}')

    # Extract scores for the methods
    scores = [systemEval[method] for method in methods]
    # Use rankdata to handle ties in scoring
    ranks2 = rankdata([-score for score in scores], method='average')  # Negative scores for descending order

    
    # Calculate Spearman's Rank Correlation Coefficient
    spearman_correlation:float
    spearman_p_value:float
    kendall_correlation:float
    kendall_p_value:float
    spearman_correlation, spearman_p_value = spearmanr(ranks1, ranks2)
    kendall_correlation, kendall_p_value = compatible_kendalltau(ranks1, ranks2)

    return CorrelationStats(spearman_correlation=spearman_correlation, kendall_correlation=kendall_correlation)

```


We manually transcribed the official leaderboards from the respective TREC Overview notebooks as follows.

```python
official_DL19_leaderboard:Dict[str,float] = {
                        "idst_bert_p1": 1,
                        "idst_bert_p2": 2,
                        "idst_bert_p3": 3,
                        "p_exp_rm3_bert": 4,
                        "p_bert": 5,
                        "ids_bert_pr2": 6,
                        "ids_bert_pr1": 7,
                        "p_exp_bert": 8,
                        "test1": 9,
                        "TUA1-1": 10,
                        "runid4": 11,
                        "runid3": 12,
                        "TUW19-p3-f": 13,
                        "TUW19-p1-f": 14,
                        "TUW19-p3-re": 15,
                        "TUW19-p1-re": 16,
                        "TUW19-p2-f": 17,
                        "ICT-BERT2": 18,
                        "srchvrs_ps_run2": 19,
                        "TUW19-p2-re": 20,
                        "ICT-CKNRM_B": 21,
                        "ms_duet_passage": 22,
                        "ICT-CKNRM_B50": 23,
                        "srchvrs_ps_run3": 24,
                        "bm25tuned_prf_p": 25,
                        "bm25base_ax_p": 26,
                        "bm25tuned_ax_p": 27,
                        "bm25base_prf_p": 28,
                        "runid2": 29,
                        "runid5": 30,
                        "bm25tuned_rm3_p": 31,
                        "bm25base_rm3_p": 32,
                        "bm25base_p": 33,
                        "srchvrs_ps_run1": 34,
                        "bm25tuned_p": 35,
                        "UNH_bm25": 36
                    }
```

```python
official_CarY3_leaderboard:Dict[str,float] = { 
                                              "dangnt-nlp": 1
                                            , "ReRnak3_BERT": 2
                                            , "ReRnak2_BERT": 3
                                            , "IRIT1" : 5
                                            , "IRIT2" : 5
                                            , "IRIT3" : 5
                                            , "ECNU_BM25_1" : 7.5
                                            , "ECNU_ReRank1" : 7.5
                                            , "Bert-ConvKNRM-50" : 9
                                            , "bm25-populated" : 10
                                            , "UNH-bm25-ecmpsg" : 11
                                            , "Bert-DRMMTKS" : 12
                                            , "UvABM25RM3" : 13
                                            , "UvABottomUpChangeOrder" : 14
                                            , "UvABottomUp2" : 15
                                            , "ICT-DRMMTKS" : 16
}
```

We compare to the leaderboard produced by the Original EXAM method (Sander \& Dietz, 2021)

```python
origExamLeaderboard:Dict[str,float]  = { 
                        "ReRnak2_BERT": 1
                      , "dangnt-nlp": 2
                      , "Bert-DRMMTKS": 3
                      , "IRIT2": 4
                      , "ReRnak3_BERT": 5
                      , "Bert-ConvKNRM-50": 6
                      , "IRIT1": 7
                      , "bm25-populated": 8
                      , "IRIT3": 9
                      , "UNH-bm25-ecmpsg": 10
                      , "ECNU_BM25_1": 11
                      , "ECNU_ReRank1": 12
                      , "ICT-DRMMTKS": 13
                      , "UvABottomUpChangeOrder": 14
                      , "UvABM25RM3": 15
                      , "UvABottomUp2": 16
}
```

### Cohen's kappa Inter-annotator Agreement

Cohen's kappa inter-annotator agreement: per-passage, do the relevance labels predicted with our methods agree with relevance judgments according to the official TREC assessors?

We use binary agreements based on a confusion matrix. For graded relevance, we either use an exact match with the grade, or we merge different grades. As an example, Figure 5, bottom merges predicted relevance grades 4+5 into the relevant class, to be compared to judgment grades 1+2+3. 

```python
    def cohen_kappa(self)->float:
        total = self.all() # self.predictedRelevant + self.predictedButNotRelevant +
                            #  self.notPredictedButRelevant + self.notPredictedNotRelevant
        po = frac(self.predictedRelevant + self.notPredictedNotRelevant,  total)

        pyes = frac(self.predictedRelevant + self.predictedButNotRelevant, total)
        pno = frac(self.notPredictedButRelevant + self.notPredictedNotRelevant,  total)
        pyes_rated = frac(self.predictedRelevant + self.notPredictedButRelevant,  total)
        pno_rated = frac(self.predictedButNotRelevant + self.notPredictedNotRelevant , total)

        pe = (pyes * pyes_rated) + (pno * pno_rated)
        if(pe==1.0):
            print(f'Warning, pe=1.0, can\'t compute kappa. {self}')
            return 0.0
        
        kappa = (po - pe) / (1 - pe)
        return kappa
```

