import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import os
from huggingface_hub import login
login(token="") # input your huggingface token here


def get_minicpm_response(folder_name, system_prompt, user_prompt, model_name, turn_index, context, bool_list, context_index, language):
    candidate_image_names = ["1.png", "2.png", "3.png"]
    image_name_path = os.path.join(folder_name, candidate_image_names[turn_index])
    if context == []:
        prompt = f"{system_prompt}\n{user_prompt}"
    else:
        prompt = user_prompt
    context.append({
        'role': 'user',
        'content': [Image.open(image_name_path).convert('RGB'), prompt]
    })

    tmp_folder_list = [["Static", "Dynamic"],
                       ["base", "No_CoT", "One-shot", "Consistency", "Build_only", "Hint_only",
                        "Self-feedback_less_context", "Retrieve_real_image_less_context", "Multi-view",
                        "Multilinguality", "Multi-questions"]]
    folder_path_with_prefix = ''
    for i in range(len(bool_list)):
        tmp_folder_name = tmp_folder_list[i][bool_list[i]]
        folder_path_with_prefix = os.path.join(folder_path_with_prefix, tmp_folder_name)
        os.makedirs(folder_path_with_prefix, exist_ok=True)

    parts = folder_name.split('/')
    folder_type = f'evaluation_{model_name}'
    group_folder_name = parts[-1]
    os.makedirs(os.path.join(folder_path_with_prefix, folder_type), exist_ok=True)
    group_folder_path = os.path.join(folder_path_with_prefix, folder_type, group_folder_name)
    os.makedirs(group_folder_path, exist_ok=True)

    file_path = os.path.join(group_folder_path, f'output_of_turn_{turn_index}_try_{context_index}.txt')
    if os.path.exists(file_path):
        return context

    if model_name != "MiniCPM-V-2_6":
        raise ValueError

    model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
                                      attn_implementation='sdpa',
                                      torch_dtype=torch.bfloat16)  # sdpa or flash_attention_2, no eager
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)
    response = model.chat(
        image=None,
        msgs=context,
        tokenizer=tokenizer,
        max_new_tokens=1000,
        sampling=False,
    )
    with open(file_path, 'w') as outfile:
        outfile.write(response)
    return context