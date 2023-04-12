"""

"""
import json
import os
import re
from multiprocessing import Pool

import numpy as np
import tqdm

import fire

from llm_model import LLM

import numpy as np


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def generate_questions(model, sentence):
    instruction = 'Ask five questions about follow article.'
    input_model = sentence
    response = model.evaluate(instruction=instruction, input=input_model)[0]
    questions = response.split('\n')
    questions = [c[3:] for c in questions]
    return questions


def generate_answer(model, question, sentence):
    prompt = f'Answer the question according to the text. Question: {question}. Text: {sentence}'
    response = model.evaluate(instruction=prompt)
    return response

def get_LLM(
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "tloen/alpaca-lora-7b",
        prompt_template: str = "",  # The prompt template to use, will default to alpaca.
):
    model = LLM(
        load_8bit,
        base_model,
        lora_weights,
        prompt_template
    )
    return model


def main(
        document_file='',
        base_model='decapoda-research/llama-7b-hf',
        lora_weights: str = "tloen/alpaca-lora-7b",
        load_8bit: bool = False,
        worker_number: int = 1
):
    # load document_file
    with open(document_file, 'r') as f:
        documents = f.readlines()
    
    print()
    sentence_number = len(documents)
    print(f"{sentence_number} sentence need to be process.")

    per_worker = int(np.ceil(sentence_number / worker_number))
    
    # p = Pool(worker_number)

    # start worker
    for i in range(worker_number):
        start_ind = per_worker * i
        end_ind = min(per_worker * (i+1), sentence_number)
        # p.apply_async(worker, args=(documents[start_ind:end_ind], start_ind,
        #        i, base_model, lora_weights, load_8bit))
        worker(documents=documents[start_ind:end_ind], start_id=start_ind,
               worker_ids=i, base_model=base_model, lora_weights=lora_weights, load_8bit=load_8bit)
    # p.close()
    # p.join()

    # merge results
    merged_results = []
    for i in range(worker_number):
        with open(os.path.join('./worker_tmp', 'result_{:02d}.json'.format(i)), 'r') as f:
            merged_results += json.load(f)
    
    print('Total qa number: ', len(merged_results))
    with open(os.path.splitext(document_file)[0] + '.json', 'w') as f:
        json.dump(merged_results, f, indent=4)

def worker(
    documents,
    start_id,
    worker_ids,        
    base_model,
    lora_weights,
    load_8bit,
):
    os.environ['CUDA_VISIBLE_DEVICES']=str(worker_ids)
    # build LLM
    model = get_LLM(base_model=base_model, lora_weights=lora_weights, load_8bit=load_8bit)

    qa_list = []
    start_id -= 1
    for sentence in tqdm.tqdm(documents):
        # get_question
        start_id += 1
        try:
            questions = generate_questions(model, sentence)
        except:
            continue

        for q in questions:
            # filter the too short question
            if len(q.split()) < 3:
                continue
            # get_answer
            answer = generate_answer(model, q, sentence)
            qa_list.append({"instruction": q, "output": answer[0], "input": "", "source_line": start_id})
        
    # save results
    os.makedirs('./worker_tmp', exist_ok=True)
    with open(os.path.join('./worker_tmp', 'result_{:02d}.json'.format(worker_ids)), 'w') as f:
        json.dump(qa_list, f, indent=4)


if __name__ == "__main__":
    fire.Fire(main)
