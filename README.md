# Appendix for: Pencils Down! Evaluating Information Content with FLAN-T5-EXAM


## Errata

In Figure 2, due to a drag-and-drop error the methods "UvABottomUpCha." and "UvABM25RM3" were flipped. The data was correctly presented, just the sort order was incorrect. Updated figure below.

![Figure 2: Comparing TREC CAR Y3 leaderboards.](car-y3-leaderboard.png)




# Detailed Experimental Results

## Generated questions

* CAR:  [car-y3-genq.jsonl.gz](car-y3-questions/car-y3-genq.jsonl.gz)
* DL19: [dl19-genq.jsonl.gz](dl19-questions/dl19-genq.jsonl.gz)
* DL20: [dl20-genq.jsonl.gz](dl20-questions/dl20-genq.jsonl.gz)


For CAR we also use corresponding questions from the [TQA dataset](https://allenai.org/data/tqa)

## Results on CAR Y3: 

 EXAM, n-EXAM and standard errors for each column in Table 3):

   *  [GenQ Exam Cover](car-results/leaderboard-car-genq-self-rating-4.md) 
   *  [GenQ Exam Qrels](car-results//qrel-leaderboard-car-genq-self-rating.md)
   *  [TQA Exam Cover](car-results/leaderboard-car-tqa-cc-verified.md) 
   * [Squad2-fine-tuned](car-results/leaderboard-car-tqa-squad2-verified.md)    -- omitted -- 

 Also available as *TSV format in folder `./car-results`

 [./car-results/leaderboard-car.gnumeric](./car-results/leaderboard-car.gnumeric)  collected results for EXAM including the plot for Figure 2 (sheet: "pretty").

## Results on TREC DL

EXAM, n-EXAM and standard errors

  * [GenQ ExamCover DL19](dl19-results/EXAM-Cover-leaderboard-dl19-subqueries-self-rating-3.md)
  * [GenQ ExamQrels DL19](dl19-results/exam-qrels-leaderboard-dl19-qrels-genq-self-rating-3.md)  -  [qrels file](dl19-results/dl19exam-3.qrels)
  
  Also avaulable as *TSV in folder `./dl19-results`
  
  
  * [GenQ ExamCover DL20](dl20-results/EXAM-Cover-leaderboard-dl20-subqueries-self-rating-4.md)
  * [GenQ ExamQrels DL20](dl20-results/exam-qrels-leaderboard-dl20-qrels-genq-self-rating-4.md)    -  [qrels file](dl20-results/dl20-exam-4.qrels)
  Also avaulable as *TSV in folder `./dl20-results`
  

Results from relevance-grading baslines (Sun, Fag, Thom):  [results-relevance-grading.md](results-relevance-grading.md)



## Test Collections

In the experimental evaluation, we use the following test collections:

* TREC CAR Y3:   <http://trec-car.cs.unh.edu/datareleases/>
  *  qrels and runs: [trec-car-runs-and-eval.tar.gz](http://trec-car.cs.unh.edu/results-Y3/trec-car-y3-runs-and-eval.tar.gz)

* TREC DL 19:   <https://trec.nist.gov/data/deep2019.html>
   * qrels [NIST judgments for the Passage Ranking task](https://trec.nist.gov/data/deep/2019qrels-pass.txt)
   * Runs  (l/p  via TREC organizers) [Deep Learning Track Passage Ranking Task Submission Files](https://trec.nist.gov/results/trec28/deep.passages.input.html)

* TREC DL 20:   <https://trec.nist.gov/data/deep2020.html> 
   * Qrels: [NIST judgments for the Passage Ranking task](https://trec.nist.gov/data/deep/2020qrels-pass.txt)
   * Runs  (l/p  via TREC organizers) [Deep Learning Track Passage Ranking Task Submission Files](https://trec.nist.gov/results/trec29/deep.passages.input.html)


## Prompts

### EXAM Question Generation Promps

The question generation prompts should reflect the goals of the IR tasks, and hence are data set specific.

* EXAM Question Generation for TREC CAR Y3:   [exam-qgen-CAR](prompts/exam-qgen-CAR)
* EXAM QuestionGeneration for TREC DL 2019 and 2020:   [exam-qgen-DL](prompts/exam-qgen-DL)


### EXAM Grading Prompts


* EXAM Grading prompt for question answering (for answer verification below):     [exam-grading-qa](prompts/exam-grading-qa)
* EXAM Grading for Self-Rating of answerability     [exam-grading-self-rating](prompts/exam-grading-self-rating)




### Baselines: Relevance Judgment Prompts

As baselines we include prompts that directly ask the LLM to grade the relevance of a passage for a given query. We use the following prompts cited in literature:

* Faggioli et al: [Fag](prompts/Fag)   [Fag-few](prompts/Fag-few)
* Sun et al: [Sun](prompts/Sun)   [Sun-few](prompts/Sun-few)
* Thomas et al: [Thom](prompts/Thom) 




## EXAM Grading with FLAN-T5


For grading, we use the [FLAN-T5-large](https://huggingface.co/google/flan-t5-large) model with the `text2text-generation` pipeline from Hugging Face.

Details are provided in [flan-t5-pipeline.md](flan-t5-pipeline.md)

## Answer Verification for Q/A

Details are provided at [answer-verification.md](answer-verification.md)


# Experimental Data

## Evaluation Measures

More details provided in [evaluation-measures.md](evaluation-measures.md)

### Leaderboard Rank Correlation

Using the official leaderboard as  reference, we use  Spearman's rank correlation coefficient ( `spearmanr`)  and and Kendall's Tau rank correlation ( `kendalltau`) from the   `scipy.stats` package.

We manually transcribed the official leaderboards from the respective TREC Overview notebooks.


### Inter-annotator Agreement

Cohen's kappa inter-annotator agreement are computed per-passage. 

We use binary agreements based on a confusion matrix, asking: "do the relevance labels predicted with our methods agree with relevance judgments according to the official TREC assessors?"
 For graded relevance, we either use an exact match with the grade, or we merge different grades. As an example, Figure 5, bottom merges predicted relevance grades 4+5 into the relevant class, to be compared to judgment grades 1+2+3. 

We implement Cohen's kappa as follows:

```python
        pe = (pyes * pyes_rated) + (pno * pno_rated)
        kappa = (po - pe) / (1 - pe)
```


## Raw EXAM graded data

### CAR-Y3

 CAR Y3:  [CAR-graded-t5-rating-genq-tqa-rating-cc-exam-qrel-runs-result.jsonl.xz](EXAM-graded-data/CAR-graded-t5-rating-genq-tqa-rating-cc-exam-qrel-runs-result.jsonl.xz) 
 
 This file contains passages from official qrels files and the top 20 (per section-level query) of all run files submitted by participants to the TREC track.

EXAM grades for the following variations are contained (obtained with the filter criteria below):
 
*  TQA Exam Cover ( TQA questions, verification with QA and Answer Verification): 
   * `"llm": "google/flan-t5-large"`
   * `"prompt_info.prompt_class": "QuestionCompleteConcisePromptWithAnswerKey2",`
   * TQA question IDs,  format `NDQ_{number}` 


*  GenQ Exam Cover and GenQ Exam Qrels  ( Generated questions, Self-rated): 
   * `"llm": "google/flan-t5-large"`
   * `"prompt_info.prompt_class": "QuestionSelfRatedUnanswerablePromptWithChoices",`
   * question ID format `tqa2:{query_id}/{query_subtopic_id)/{md5_hash_of_question_text)`


*  Result omitted for brevity( TQA questions, Self-rated): 
   * `"llm": "google/flan-t5-large"`
   * `"prompt_info.prompt_class": "QuestionSelfRatedUnanswerablePromptWithChoices",`
   * TQA question IDs,  format `NDQ_{number}` 



EXAM grades for additional (omitted) question-answering  experiments for CAR Y3: [CAR-graded-squad2-t5-qa-tqa-exam--benchmarkY3test-exam-qrels-runs-with-text.jsonl.xz](EXAM-graded-data/CAR-graded-squad2-t5-qa-tqa-exam--benchmarkY3test-exam-qrels-runs-with-text.jsonl.xz) 
 
This file contains the same passages. Exam grades for the following variation are contained (obtained with the filter criteria below):


 * not included (TQA questions, squad2-finetuned model of FLAN-T5-large, verification with QA and Answer Verification)
   * `"llm": "sjrhuschlee/flan-t5-large-squad2"`
   * `"prompt_info.prompt_class": "QuestionCompleteConcisePromptWithAnswerKey2",`
   * TQA question IDs,  format `NDQ_{number}` 
   
   
 * not included (TQA questions, untuned model, prompt without instructions, verification with QA and Answer Verification)
   * `"llm": "google/flan-t5-large"`
   * `"prompt_info.prompt_class": "QuestionCompleteConcisePromptWithAnswerKey2",`
   * TQA question IDs,  format `NDQ_{number}` 

### TREC DL 2019/2020
 
DL19:  [DL19-graded-t5-rating-genq-exam-qrels-runs-with-text.jsonl.gz](EXAM-graded-data/DL19-graded-t5-rating-genq-exam-qrels-runs-with-text.jsonl.gz) 
DL20:  [DL20-graded-t5-rating-genq-exam-qrels-runs-with-text.jsonl.gz](EXAM-graded-data/DL20-graded-t5-rating-genq-exam-qrels-runs-with-text.jsonl.gz) 
 
 This file contains passages from official qrels files and the top 20 (per section-level query) of all run files submitted by participants to the TREC track.

EXAM grades for the following variations are contained (obtained with the filter criteria below):

*  GenQ Exam Cover and GenQ Exam Qrels  ( Generated questions, Self-rated): 
   * `"llm": "google/flan-t5-large"`
   * `"prompt_info.prompt_class": "QuestionSelfRatedUnanswerablePromptWithChoices",`
   * question ID format `tqa2:{query_id}/{md5_hash_of_question_text)`

 

