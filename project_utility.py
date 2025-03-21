import pandas as pd
from datasets import load_dataset
import evaluate
from sentence_transformers import SentenceTransformer, util
import torch
from pprint import pprint

def load_data():
    ds = load_dataset("rony/climate-change-MRC")

    train_ds = ds["train"] 
    valid_ds = ds["validation"]
    test_ds = ds["test"]

    # each is a 1-item list, so take first index
    train_ds = train_ds[0]
    valid_ds = valid_ds[0]
    test_ds = test_ds[0]

    # take the 'data' key of the dict, ignoring 'version' (there's just one)
    train_ds = train_ds['data'][0]['paragraphs']
    valid_ds = valid_ds['data'][0]['paragraphs']
    test_ds = test_ds['data'][0]['paragraphs']
    # each dataset is a list of dicts, where each list item is a context paragraph ('context' key) with qas ('qas' key) which contain questions, id, and answer

    train_df = pd.DataFrame(prepare_data(train_ds))
    print(f"{train_df.shape=}")

    valid_df = pd.DataFrame(prepare_data(valid_ds))
    print(f"{valid_df.shape=}")

    test_df = pd.DataFrame(prepare_data(test_ds))
    print(f"{test_df.shape=}")

    return train_df, valid_df, test_df


# using code from here: https://medium.com/@ajazturki10/simplifying-language-understanding-a-beginners-guide-to-question-answering-with-t5-and-pytorch-253e0d6aac54
def prepare_data(data):
    articles = []

    for paragraph in data:
        context = paragraph['context']
        for qa in paragraph['qas']:
            question = qa['question']
            id = qa['id']
            for ans in qa['answers']:
                answer = ans['text']
                answer_start = ans['answer_start']
                articles.append({'context': context, 'question': question, 'id': id, 'answer': answer, 'answer_start': answer_start})

    return articles

def evaluate_abstractive(result_df, 
                         pred_col, 
                         ref_col='answer', 
                         encoder_model='sentence-transformers/all-MiniLM-L12-v2'):
    predictions = result_df[pred_col].tolist()
    references = result_df[ref_col].tolist()

    rouge = evaluate.load('rouge')
    rouge_res = rouge.compute(predictions=predictions,
                              references=references)
    
    encoder_model = SentenceTransformer(encoder_model)
    candidate_embeddings = encoder_model.encode(predictions)
    reference_embeddings = encoder_model.encode(references)
    similarity = util.pairwise_cos_sim(candidate_embeddings, reference_embeddings)
    
    print('rouge scores:')
    pprint(rouge_res)
    print()
    print('average semantic similarity:')
    print(torch.mean(similarity))