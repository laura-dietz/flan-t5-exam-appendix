CAR, TQA questions, QA with answer verification,  flan-t5-large finetuned on Squad2 dataset.

| method | exam | +/- | exam-std | n-exam | +/- | n-exam-std | orig\_TREC\_leaderboard\_rank | orig\_EXAM\_leaderboard\_rank |  |  |  |  |  |  |  |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | --- | --- | --- | --- | --- |
| Bert-ConvKNRM | 0.169 | +/- | 0.012 | 0.515 | +/- | 0.032 |  |  |  |  |  |  |  |  |  |
| Bert-ConvKNRM-50 | 0.195 | +/- | 0.013 | 0.573 | +/- | 0.032 | 9.000 | 6.000 |  |  |  |  |  |  |  |
| Bert-DRMMTKS | 0.143 | +/- | 0.011 | 0.415 | +/- | 0.030 | 12.000 | 3.000 |  |  |  |  |  |  |  |
| ECNU\_BM25 | 0.196 | +/- | 0.013 | 0.576 | +/- | 0.032 |  |  |  |  |  |  |  |  |  |
| ECNU\_BM25\_1 | 0.195 | +/- | 0.013 | 0.578 | +/- | 0.032 | 7.500 | 11.000 |  |  |  |  |  |  |  |
| ECNU\_ReRank1 | 0.194 | +/- | 0.013 | 0.570 | +/- | 0.033 | 7.500 | 12.000 |  |  |  |  |  |  |  |
| ICT-BM25 | 0.193 | +/- | 0.013 | 0.571 | +/- | 0.033 |  |  |  |  |  |  |  |  |  |
| ICT-DRMMTKS | 0.146 | +/- | 0.012 | 0.409 | +/- | 0.030 | 16.000 | 13.000 |  |  |  |  |  |  |  |
| IRIT1 | 0.186 | +/- | 0.012 | 0.545 | +/- | 0.032 | 5.000 | 7.000 |  |  |  |  |  |  |  |
| IRIT2 | 0.186 | +/- | 0.012 | 0.545 | +/- | 0.032 | 5.000 | 4.000 |  |  |  |  |  |  |  |
| IRIT3 | 0.186 | +/- | 0.012 | 0.545 | +/- | 0.032 | 5.000 | 9.000 |  |  |  |  |  |  |  |
| ReRnak2\_BERT | 0.205 | +/- | 0.013 | 0.612 | +/- | 0.033 | 3.000 | 1.000 |  |  |  |  |  |  |  |
| ReRnak3\_BERT | 0.197 | +/- | 0.013 | 0.584 | +/- | 0.033 | 2.000 | 5.000 |  |  |  |  |  |  |  |
| UNH-bm25-ecmpsg | 0.184 | +/- | 0.012 | 0.537 | +/- | 0.033 | 11.000 | 10.000 |  |  |  |  |  |  |  |
| UNH-bm25-rm | 0.186 | +/- | 0.013 | 0.546 | +/- | 0.033 |  |  |  |  |  |  |  |  |  |
| UNH-bm25-stem | 0.184 | +/- | 0.012 | 0.537 | +/- | 0.033 |  |  |  |  |  |  |  |  |  |
| UNH-dl100 | 0.184 | +/- | 0.012 | 0.537 | +/- | 0.033 |  |  |  |  |  |  |  |  |  |
| UNH-dl300 | 0.184 | +/- | 0.012 | 0.537 | +/- | 0.033 |  |  |  |  |  |  |  |  |  |
| UNH-ecn | 0.183 | +/- | 0.012 | 0.540 | +/- | 0.032 |  |  |  |  |  |  |  |  |  |
| UNH-qee | 0.194 | +/- | 0.013 | 0.571 | +/- | 0.032 |  |  |  |  |  |  |  |  |  |
| UNH-tfidf-lem | 0.184 | +/- | 0.012 | 0.537 | +/- | 0.033 |  |  |  |  |  |  |  |  |  |
| UNH-tfidf-ptsim | 0.184 | +/- | 0.012 | 0.537 | +/- | 0.033 |  |  |  |  |  |  |  |  |  |
| UNH-tfidf-stem | 0.184 | +/- | 0.012 | 0.537 | +/- | 0.033 |  |  |  |  |  |  |  |  |  |
| UvABM25RM3 | 0.079 | +/- | 0.008 | 0.235 | +/- | 0.024 | 13.000 | 15.000 |  |  |  |  |  |  |  |
| UvABottomUp1 | 0.066 | +/- | 0.008 | 0.180 | +/- | 0.021 |  |  |  |  |  |  |  |  |  |
| UvABottomUp2 | 0.074 | +/- | 0.008 | 0.210 | +/- | 0.021 | 15.000 | 16.000 |  |  |  |  |  |  |  |
| UvABottomUpChangeOrder | 0.079 | +/- | 0.008 | 0.235 | +/- | 0.024 | 14.000 | 14.000 |  |  |  |  |  |  |  |
| \_overall\_ | 0.281 | +/- | 0.015 | 1.000 | +/- | 0.000 |  |  |  |  |  |  |  |  |  |
| bm25-populated | 0.187 | +/- | 0.012 | 0.557 | +/- | 0.033 | 10.000 | 8.000 |  |  |  |  |  |  |  |
| dangnt-nlp | 0.209 | +/- | 0.013 | 0.635 | +/- | 0.034 | 1.000 | 2.000 |  |  |  |  |  |  |  |
| spearman | 0.838 |  |  | 0.850 |  |  |  |  |  |  |  |  |  |  |  |
| kendall | 0.687 |  |  | 0.704 |  |  |  |  |  |  |  |  |  |  |  |
| EXAM scores produced with GradeFilter(model\_name='sjrhuschlee/flan-t5-large-squad2', prompt\_class='QuestionCompleteConcisePromptWithAnswerKey2', is\_self\_rated=None, min\_self\_rating=None, question\_set='tqa') |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| min\_rating | None |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

