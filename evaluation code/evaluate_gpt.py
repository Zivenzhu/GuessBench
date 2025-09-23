import base64
import os
import json
import time
from openai import AzureOpenAI

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


client = AzureOpenAI(
    api_version="2025-01-01-preview",
    api_key="", # input your api key here    
    azure_endpoint="" # input your endpoint here    
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_images_in_folder(folder_name, turn_index):
    image_files = ["1.png", "2.png", "3.png"]
    name = image_files[turn_index]
    file_path = os.path.join(folder_name, name)
    base64_img = encode_image(file_path)
    return base64_img


def google_image_search(query, num_images=5):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # 无界面模式
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(options=options)

    try:
        # 打开 Google 图片
        driver.get("https://www.google.com/imghp")

        # 输入查询关键词并搜索
        search_box = driver.find_element(By.NAME, "q")
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)

        # 等待图片加载
        WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.H8Rx8c img"))
        )
        # 滚动页面，加载更多图片
        for _ in range(3):
            driver.execute_script("window.scrollBy(0, 1000);")
            time.sleep(1)

        # 获取图片
        images = driver.find_elements(By.CSS_SELECTOR, "div.H8Rx8c img")[:num_images]
        img_urls = [img.get_attribute("src") for img in images if img.get_attribute("src")]

        return img_urls

    finally:
        driver.quit()


def get_answer(guess):
    client = AzureOpenAI(
        api_version="2025-01-01-preview",
        api_key="",  # input your api key here
        azure_endpoint=""  # input your endpoint here  
    )

    completion = client.chat.completions.create(
        model="gpt4o",  # "gpt4o", "gpt4o-mini"
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
    return extrated_answer

def get_gpt_response(folder_name, system_prompt, user_prompt, model_name, turn_index, context, bool_list, context_index, language=""):
    if bool_list[1] != 8:
        base64_img = process_images_in_folder(folder_name, turn_index)
    context.append({
            "role": "system",
            "content": system_prompt
    })
    if bool_list[1] == 2:
        all_hint = []
        shot_demo = [
                {
                    "type": "text",
                    "text": "\nI will give you an example to help you guess the build. Here is an example:"
                }
                ]
        for i in range(turn_index+1):
            with open(f"shot1/Hint_{i+1}.txt", "r") as file:
                all_hint.append(file.read())
            shot_demo.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{process_images_in_folder('shot1', i)}"
                    }
                })
            shot_demo.append({
                    "type": "text",
                    "text": f"Hint_{i+1}: {all_hint[i]}"
                })
        with open(f"shot1/Answer.txt", "r") as file:
            answer = file.read()
        shot_demo.append({
                    "type": "text",
                    "text": f"Answer: The build represents {answer}."
                })

        context.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_img}"
                    }
                },
                {
                    "type": "text",
                    "text": f"Your Task:\n{user_prompt}"
                }
            ] + shot_demo
        })
    elif bool_list[1] == 5:
        context.append({
                "role": "user",
                 "content": [
                     {
                        "type": "text",
                        "text": user_prompt
                     }
                ]
            })
    elif bool_list[1] == 8:
        context.append({
                "role": "user",
                 "content": [
                     {
                         "type": "image_url",
                         "image_url": {
                             "url": f"data:image/jpeg;base64,{encode_image(os.path.join(folder_name, str(turn_index + 1), '1.png'))}"
                         }
                     },
                     {
                         "type": "image_url",
                         "image_url": {
                             "url": f"data:image/jpeg;base64,{encode_image(os.path.join(folder_name, str(turn_index + 1), '2.png'))}"
                         }
                     },
                     {
                         "type": "image_url",
                         "image_url": {
                             "url": f"data:image/jpeg;base64,{encode_image(os.path.join(folder_name, str(turn_index + 1), '3.png'))}"
                         }
                     },
                     {
                        "type": "text",
                        "text": user_prompt
                     }
                ]
            })
    else:
        context.append({
                "role": "user",
                 "content": [
                     {
                         "type": "image_url",
                         "image_url": {
                             "url": f"data:image/jpeg;base64,{base64_img}"
                         }
                     },
                     {
                        "type": "text",
                        "text": user_prompt
                     }
                ]
            })



    tmp_folder_list = [["Static", "Dynamic"],
                       ["base", "No_CoT", "One-shot", "Consistency", "Build_only", "Hint_only", "Self-feedback_less_context", "Retrieve_real_image_less_context", "Multi-view", "Multilinguality", "Multi-questions"]]
    folder_path_with_prefix = ''
    for i in range(len(bool_list)):
        tmp_folder_name = tmp_folder_list[i][bool_list[i]]
        folder_path_with_prefix = os.path.join(folder_path_with_prefix, tmp_folder_name)
        os.makedirs(folder_path_with_prefix, exist_ok=True)
    if bool_list[1] == 9:
        folder_path_with_prefix = os.path.join(folder_path_with_prefix, language)
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


    if bool_list[1] != 3:
        completion = client.chat.completions.create(
            model=model_name,  # "gpt4o", "gpt4o-mini"
            # modalities=["text"],
            messages=context,
            max_tokens=1000,
            temperature=0.0,
            top_p=1.0,
            seed=42,
        )
        response = json.loads(completion.json())["choices"][0]["message"]["content"]
        with open(file_path, 'w') as outfile:
            outfile.write(response)
        time.sleep(2)

        if bool_list[1] == 6:

            tmp_response = json.loads(completion.json())["choices"][0]["message"]["content"]
            new_file_path = os.path.join(group_folder_path,
                                         f'output_of_turn_{turn_index}_try_{context_index}_feedback_0.txt')
            with open(new_file_path, 'w') as outfile:
                outfile.write(tmp_response)

            for retry_time in range(3):
                reflection_prompt = f"""Here is your response: {tmp_response}
Please review your answer to check if it aligns with the given hint. If it does, you MUST ONLY output "$Well done!$" Otherwise, provide your improved guess."""
                tmp_context = context + [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": reflection_prompt
                        }
                    ]
                }]
                completion = client.chat.completions.create(
                    model=model_name,  # "gpt4o", "gpt4o-mini"
                    # modalities=["text"],
                    messages=tmp_context,
                    max_tokens=1000,
                    temperature=0.0,
                    top_p=1.0,
                    seed=42,
                )
                tmp_response = json.loads(completion.json())["choices"][0]["message"]["content"]
                if ("$Well done!$" in tmp_response) and ("""If it does, you MUST ONLY output "$Well done!$" Otherwise, provide your improved guess.""" not in tmp_response):
                    break
                else:
                    response = tmp_response
                    with open(file_path, 'w') as outfile:
                        outfile.write(response)
                    time.sleep(2)
                    new_file_path = os.path.join(group_folder_path, f'output_of_turn_{turn_index}_try_{context_index}_feedback_{retry_time+1}.txt')
                    with open(new_file_path, 'w') as outfile:
                        outfile.write(response)

        elif bool_list[1] == 7:
            tmp_response = json.loads(completion.json())["choices"][0]["message"]["content"]
            new_file_path = os.path.join(group_folder_path,
                                         f'output_of_turn_{turn_index}_try_{context_index}_real-image_0.txt')
            with open(new_file_path, 'w') as outfile:
                outfile.write(tmp_response)


            for retry_time in range(3):
                extracted_guess = get_answer(tmp_response).split("Answer:", 1)[-1]
                print(extracted_guess)
                reflection_prompt = f"""Previously, your guess was: {extracted_guess}.

Now, I have searched for a real image of {extracted_guess} from Google Images and attached it above. Please compare the provided Minecraft build with the real image of your guess.

If they match, you MUST ONLY output: "$Well done!$". Otherwise, provide your improved guess."""
                real_image = google_image_search(extracted_guess, 1)[0]
                tmp_context = context + [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": reflection_prompt
                        },
                        {
                            "type": "text",
                            "text": "Here is the provided Minecraft build."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_img}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Here is the real image of your guess."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": real_image
                            }
                        },
                    ]
                }]
                completion = client.chat.completions.create(
                    model=model_name,  # "gpt4o", "gpt4o-mini"
                    # modalities=["text"],
                    messages=tmp_context,
                    max_tokens=1000,
                    temperature=0.0,
                    top_p=1.0,
                    seed=42,
                )
                time.sleep(2)
                tmp_response = json.loads(completion.json())["choices"][0]["message"]["content"]
                print(tmp_response)
                if ("$Well done!$" in tmp_response) and (
                        """If they match, you MUST ONLY output: "$Well done!$". Otherwise, provide your improved guess.""" not in tmp_response):
                    break
                else:
                    response = tmp_response
                    with open(file_path, 'w') as outfile:
                        outfile.write(response)
                    time.sleep(2)
                    new_file_path = os.path.join(group_folder_path,
                                                 f'output_of_turn_{turn_index}_try_{context_index}_real-image_{retry_time + 1}.txt')
                    with open(new_file_path, 'w') as outfile:
                        outfile.write(response)



    else:
        completion = client.chat.completions.create(
            model=model_name,  # "gpt4o", "gpt4o-mini"
            messages=context,
            max_tokens=1000,
        )
        response = json.loads(completion.json())["choices"][0]["message"]["content"]
        with open(file_path, 'w') as outfile:
            outfile.write(response)
        time.sleep(2)


    return context
