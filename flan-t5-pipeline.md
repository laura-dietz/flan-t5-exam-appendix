#  Prompting the FLAN-T5 model

We prompt the FLAN-T5-large model  via `text2text-generation` pipeline from Hugging Face.

Each question and passage is converted into a prompt for grading using a call to `generate_prompt_with_context_no_choices`.  For multiple-choice questions, only the question is given and choices are held out from the process. We found that this curbs the model's tendency to answer question from memory.

To respect the  512 maximum token length, we truncate given context, ensuring that the prompt and question remains intact.


```python


class Text2TextPipeline():
    """QA Pipeline for text2text based question answering"""

    def __init__(self, model_name:str):
        """Run given prompts and collect response.
                Example:  modelName = 'google/flan-t5-large'
           """
        self.question_batchSize = 100 # batchSize
    
        # Initialize the tokenizer and model
        self.modelName = model_name
        self.model = T5ForConditionalGeneration.from_pretrained(self.modelName)
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelName)

        print(f"Text2Text model config: { self.model.config}")
        print("maxBatchSize",computeMaxBatchSize(self.model.config))
        self.max_token_len = 512

        # Create a Hugging Face pipeline
        self.t5_pipeline_qa = pipeline('text2text-generation', model=self.model, tokenizer=self.tokenizer, device=device, batch_size=BATCH_SIZE, use_fast=True)

    def exp_modelName(self)->str:
        return self.modelName

    def batchChunker(self, iterable):
        iterator = iter(iterable)
        while True:
            batch = list(itertools.islice(iterator, self.question_batchSize))
            if not batch or len(batch)<1:
                break
            yield batch


    def chunkingBatchAnswerQuestions(self, questions:List[QuestionPrompt],  paragraph_txt:str)->List[Tuple[QuestionPrompt, str]]:
            """Run question answering over batches of questions, and tuples it up with the answers"""
            promptGenerator=lambda qpc: qpc.generate_prompt_with_context_no_choices(paragraph_txt, model_tokenizer = self.tokenizer, max_token_len = self.max_token_len)

            def processBatch(qpcs:List[QuestionPrompt])->Iterable[Tuple[QuestionPrompt, str]]:
                """Prepare a batch for question answering, tuple it up with the answers"""
                prompts = [promptGenerator(qpc) for qpc in qpcs]
                
                outputs = self.t5_pipeline_qa(prompts, max_length=MAX_TOKEN_LEN, num_beams=5, early_stopping=True)
                answers:List[str] = [output['generated_text']  for output in outputs]
                return zip(qpcs, answers, strict=True)

            return list(itertools.chain.from_iterable(
                        (processBatch(batch) for batch in self.batchChunker(questions)) 
                        )) 

```
