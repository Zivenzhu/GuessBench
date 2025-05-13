from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import json
from PIL import Image
import torch


def get_qwen_response(folder_name, system_prompt, user_prompt, model_name, turn_index, context, bool_list, context_index):
    candidate_image_names = ["1.png", "2.png", "3.png"]
    image_name_path = os.path.join(folder_name, candidate_image_names[turn_index])

    context.append({
            "role": "system",
            "content": system_prompt
    })
    context.append({
            "role": "user",
             "content": [
                 {
                     "type": "image",
                     "image": Image.open(image_name_path).convert('RGB')
                 },
                 {
                    "type": "text",
                    "text": user_prompt
                 }
            ]
        })



    tmp_folder_list = [["Static", "Dynamic"],
                       ["base", "No_CoT", "One-shot", "Consistency"]]
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

    if model_name == "Qwen2.5-VL-72B-Instruct-AWQ":
        model_path = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
    elif model_name == "Qwen2.5-VL-7B-Instruct":
        model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    elif model_name == "Qwen2.5-VL-72B-Instruct":
        model_path = "Qwen/Qwen2.5-VL-72B-Instruct"
    else:
        raise ValueError

    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )


    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

    # Preparation for inference
    text = processor.apply_chat_template(
        context, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(context)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    if bool_list[1] != 3:
        generated_ids = model.generate(**inputs, max_new_tokens=1000,
                                       do_sample=False)
    else:
        generated_ids = model.generate(**inputs, max_new_tokens=1000,
                                       do_sample=True)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    response = output_text[0]
    with open(file_path, 'w') as outfile:
        outfile.write(response)
    return context
