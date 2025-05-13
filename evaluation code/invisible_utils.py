from openai import AzureOpenAI
import os
import json
import time
from collections import Counter
from evaluate_gpt import get_gpt_response
from evaluate_gemini import get_gemini_response
from evaluate_qwen import get_qwen_response
from evaluate_internvl import get_intervl_response
from evaluate_minicpm import get_minicpm_response


tmp_folder_list = [["Static", "Dynamic"],
                   ["base", "No_CoT", "One-shot", "Consistency", "Build_only", "Hint_only", "Self-feedback_less_context", "Retrieve_real_image_less_context", "Multi-view", "Multilinguality", "Multi-questions"]]


def get_answer(folder_name, model_name, turn_index, bool_list, try_index, language=""):
    global tmp_folder_list
    folder_path_with_prefix = ''
    for i in range(len(bool_list)):
        tmp_folder_name = tmp_folder_list[i][bool_list[i]]
        folder_path_with_prefix = os.path.join(folder_path_with_prefix, tmp_folder_name)
        os.makedirs(folder_path_with_prefix, exist_ok=True)
    if bool_list[1] == 9:
        folder_path_with_prefix = os.path.join(folder_path_with_prefix, language)
        os.makedirs(folder_path_with_prefix, exist_ok=True)

    parts = folder_name.split('/')
    folder_type = f'evaluation_{model_name}_extracted_answer'
    group_folder_name = parts[-1]
    os.makedirs(os.path.join(folder_path_with_prefix, folder_type), exist_ok=True)
    group_folder_path = os.path.join(folder_path_with_prefix, folder_type, group_folder_name)
    os.makedirs(group_folder_path, exist_ok=True)

    extracted_answer_file_path = os.path.join(group_folder_path, f'extracted_answer_of_turn_{turn_index}_try_{try_index}.txt')
    if os.path.exists(extracted_answer_file_path):
        with open(extracted_answer_file_path, "r") as file:
            extracted_answer = file.read()
        return extracted_answer

    response_file_path = os.path.join(folder_path_with_prefix, f'evaluation_{model_name}',
                                      group_folder_name, f'output_of_turn_{turn_index}_try_{try_index}.txt')
    with open(response_file_path, "r") as file:
        guess = file.read()

    client = AzureOpenAI(
        api_version="2025-01-01-preview",
        api_key="", # input your api key here
        azure_endpoint="https://tsvetshop.openai.azure.com/"
    )

    completion = client.chat.completions.create(
        model="gpt4o", # "gpt4o", "gpt4o-mini"
        modalities=["text"],
        messages=[
            {
                "role": "user",
                "content": f"""I will give you a language model's response in a 'Guess the Build' game. Identify and extract the model's guessed answer. Output it in the format: 'Answer: [extracted answer]'. If no answer is given in the response, output 'No Answering'.

Language model's response: {guess}
"""
            }
        ]
    )

    extrated_answer = json.loads(completion.json())["choices"][0]["message"]["content"]
    time.sleep(1)
    with open(extracted_answer_file_path, "w") as file:
        file.write(extrated_answer)
    return extrated_answer


def get_result_folder(model_name, bool_list, language=""):
    global tmp_folder_list
    folder_path_with_prefix = ''
    for i in range(len(bool_list)):
        tmp_folder_name = tmp_folder_list[i][bool_list[i]]
        folder_path_with_prefix = os.path.join(folder_path_with_prefix, tmp_folder_name)
        os.makedirs(folder_path_with_prefix, exist_ok=True)
    if bool_list[1] == 9:
        folder_path_with_prefix = os.path.join(folder_path_with_prefix, language)
        os.makedirs(folder_path_with_prefix, exist_ok=True)

    folder_type = f'evaluation_{model_name}_extracted_answer'
    os.makedirs(os.path.join(folder_path_with_prefix, folder_type), exist_ok=True)
    group_folder_path = os.path.join(folder_path_with_prefix, folder_type)
    os.makedirs(group_folder_path, exist_ok=True)

    return group_folder_path

def get_function(model_name):
    if "gpt4o" in model_name:
        return get_gpt_response
    elif "gemini-2.0-flash-001" in model_name:
        return get_gemini_response
    elif "Qwen2.5-VL" in model_name:
        return get_qwen_response
    elif "InternVL2_5-78B-MPO" in model_name:
        return get_intervl_response
    elif "MiniCPM-V-2_6" in model_name:
        return get_minicpm_response
    else:
        raise NotImplementedError

def get_most_frequent_value_and_index(tmp_answers):
    counter = Counter(tmp_answers)
    most_frequent = max(counter, key=counter.get)
    max_index = max(i for i, val in enumerate(tmp_answers) if val == most_frequent)
    return most_frequent, max_index


