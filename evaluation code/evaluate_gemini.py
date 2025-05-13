import random
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, SafetySetting, HarmCategory, HarmBlockThreshold, Part, Image
import os
import time

def resize_image(img):
    max_size = 1024
    if max(img.size) > max_size:
        scale = max_size / max(img.size)
        new_size = tuple(int(dim * scale) for dim in img.size)
        img = img.resize(new_size, Image.LANCZOS)
    # img = img.convert('RGB')
    return img



def get_gemini_response(folder_name, system_prompt, user_prompt, model_name, turn_index, context, bool_list):
    candidate_image_names = ["1.png", "2.png", "3.png"]
    image_name = candidate_image_names[turn_index]

    context.append(Part.from_image(Image.load_from_file(os.path.join(folder_name, image_name))))
    context.append(user_prompt)


    tmp_folder_list = [["Static", "Dynamic"],
                       ["base", "No_CoT"]]
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

    file_path = os.path.join(group_folder_path, f'output_of_turn_{turn_index}.txt')
    if os.path.exists(file_path):
        return context


    project_id = "" # input your project id here
    location = "" # input your location here
    vertexai.init(project=project_id, location=location)

    gemini_model = GenerativeModel(model_name=model_name, system_instruction=system_prompt)
    generationConfig = GenerationConfig(candidate_count=1,
            max_output_tokens=1000,
            temperature=0.0,
            top_p=1.0)

    safety_config = [
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_UNSPECIFIED,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
    ]

    response = gemini_model.generate_content(context,
                generation_config = generationConfig, safety_settings = safety_config).text

    with open(os.path.join(group_folder_path, f'output_of_turn_{turn_index}.txt'), 'w') as outfile:
        outfile.write(response)
    time.sleep(2)

    return context
