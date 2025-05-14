import os
from natsort import natsorted
import invisible_utils as utils
import json
import re

def remove_trailing_period(answer):
    return re.sub(r"[。．｡.\u3002]$", "", answer)


system_prompt = f"""You are an AI designed to play the "Guess the Build" game from the Hypixel server in Minecraft. Your task is to analyze a player's build and accurately guess what it represents. The build is constructed using Minecraft blocks, and it can depict objects, animals, structures, abstract concepts, or any other recognizable entity.

Consider common themes, contextual clues, and typical constructions in Minecraft. If a build is unclear, make educated guesses based on possible interpretations. Your goal is to guess as accurately and efficiently as possible.

Avoid overly broad answers—be specific when possible. If multiple plausible answers exist, prioritize the most likely one based on common builds in the game."""



asking_prompts = ["""Look at the provided image of a Minecraft build and the corresponding hint. Based on its structure, shape, materials, and the given hint, determine what it represents. Consider common objects, animals, buildings, abstract concepts, or any other recognizable entities typically built in Minecraft.

You can use chain of thought reasoning to arrive at the best possible guess.""",
                  """Look at the provided image of a Minecraft build and the corresponding hint. Based on its structure, shape, materials, and the given hint, determine what it represents. Consider common objects, animals, buildings, abstract concepts, or any other recognizable entities typically built in Minecraft.
                  
You are not allowed to use chain-of-thought reasoning. You must output the final answer directly.""",
                """Look at the provided image of a Minecraft build. Based on its structure, shape, materials, determine what it represents. Consider common objects, animals, buildings, abstract concepts, or any other recognizable entities typically built in Minecraft.

You can use chain of thought reasoning to arrive at the best possible guess.""",
                  """Look at the provided hint and determine what it represents. Based only on the hint, guess the most likely answer. Consider common objects, animals, buildings, abstract concepts, or any other recognizable entities.
                  
You can use chain of thought reasoning to arrive at the best possible guess.""",
                  """Look at the provided images of three different angles of the same Minecraft build and the corresponding hint. Based on its structure, shape, materials, and the given hint, determine what it represents. Consider common objects, animals, buildings, abstract concepts, or any other recognizable entities typically built in Minecraft.

You can use chain of thought reasoning to arrive at the best possible guess."""

                  ]


revision_prompts = ["""This guess is incorrect. Now, you will be provided with a more complete version of the build along with additional hints. Analyze the new information carefully and make a revised guess based on the updated build and hints.

You can use chain of thought reasoning to improve your guess.""",
                    """This guess is incorrect. Now, you will be provided with a more complete version of the build along with additional hints. Analyze the new information carefully and make a revised guess based on the updated build and hints.

You are not allowed to use chain-of-thought reasoning. You must output the final answer directly.""",
                    """This guess is incorrect. Now, you will be provided with a more complete version of the build. Analyze the new information carefully and make a revised guess based on the updated build.

You can use chain of thought reasoning to improve your guess.""",
                    """This guess is incorrect. Now, you will be provided with a more complete version of the hint. Analyze the new information carefully and make a revised guess based on the updated hint.

You can use chain of thought reasoning to improve your guess.""",
                    """This guess is incorrect. Now, you will be provided with a more complete version of the build, shown from three different angles, along with additional hints. Analyze the new information carefully and make a revised guess based on the updated build and hints.

You can use chain of thought reasoning to improve your guess."""
                    ]

# bool_lists = [[1, 6]]
bool_lists = [[1, 0], [0, 0]]
## [["Static", "Dynamic"], ["base", "No_CoT", "One-shot", "Consistency", "Build_only", "Hint_only", "Self-feedback_less_context", "Retrieve_real_image_less_context", "Multi-view", "Multilinguality"]]
# model_name = "InternVL2_5-78B-MPO"
model_name = "MiniCPM-V-2_6"
languages = ["Russian", "Chinese", "Thai", "Arabic", "Hungarian", "Marathi", "Telugu"]

for bool_list in bool_lists:
    all_accs = []
    if bool_list[1] == 8:
        base_folder = "" # input the path to the Multi-View dataset here
    else:
        base_folder = "../data"

    all_sets = natsorted(os.listdir(base_folder))
    # all_sets = ["5"]
    # for language in languages:

    language = "" # input the language here
    for tmp_set in all_sets:
        print(tmp_set)

        set_path = os.path.join(base_folder, tmp_set)

        if bool_list[1] != 9:
            with open(os.path.join(set_path, "Answer.txt"), "r") as file:
                answer = file.read()

        else:
            with open(os.path.join(set_path, f"Answer_{language}.txt"), "r") as file:
                answer = file.read()
        answer = remove_trailing_period(answer)

        if bool_list[1] == 3:
            times = 3
        else:
            times = 1
        all_context = [[] for _ in range(times)]
        max_index = 0

        accs = []
        sess = []
        if bool_list[1] == 1:
            asking_prompt = asking_prompts[1]
            revision_prompt = revision_prompts[1]
        elif bool_list[1] == 4:
            asking_prompt = asking_prompts[2]
            revision_prompt = revision_prompts[2]
        elif bool_list[1] == 5:
            asking_prompt = asking_prompts[3]
            revision_prompt = revision_prompts[3]
        elif bool_list[1] == 8:
            asking_prompt = asking_prompts[4]
            revision_prompt = revision_prompts[4]
        elif bool_list[1] == 9:
            with open(f"asking_prompt_{language}.txt", "r", encoding="utf-8") as file:
                asking_prompt = file.read()
            with open(f"revision_prompt_{language}.txt", "r", encoding="utf-8") as file:
                revision_prompt = file.read()
            with open(f"system_prompt_{language}.txt", "r", encoding="utf-8") as file:
                system_prompt = file.read()
        else:
            asking_prompt = asking_prompts[0]
            revision_prompt = revision_prompts[0]

        if bool_list[0] == 1:
        # 3 turns
            for i in range(3):
                if i == 0:
                    mision_prompt = asking_prompt
                else:
                    mision_prompt = f"""Your previous guess is: {extracted_answer}. {revision_prompt}"""

                if bool_list[1] == 8:
                    with open("indexs.json", "r") as f:
                        indexs = json.load(f)
                    hint_path = os.path.join("../data", str(indexs[int(tmp_set)-1]), f"Hint_{i+1}.txt")
                    print(hint_path)
                    with open(hint_path, "r") as file:
                        hint = file.read()
                elif bool_list[1] == 9:
                    pass
                else:
                    with open(os.path.join(set_path, f"Hint_{i+1}.txt"), "r") as file:
                        hint = file.read()


                if (bool_list[1] == 4) or (bool_list[1] == 9):
                    user_prompt = f"""{mision_prompt}"""
                else:
                    user_prompt = f"""{mision_prompt}

Hint: {hint}
"""

                if model_name != "InternVL2_5-78B-MPO-AWQ":
                    tmp_answers = []
                    previous_context = all_context[max_index].copy()
                    for context_index in range(times):
                        tmp_context = previous_context.copy()
                        all_context[context_index] = utils.get_function(model_name)(set_path, system_prompt, user_prompt, model_name, i, tmp_context, bool_list, context_index, language).copy()
                        tmp_answer = utils.get_answer(set_path, model_name, i, bool_list, context_index, language).lower()
                        tmp_answers.append(tmp_answer)
                    most_frequent_ans, max_index = utils.get_most_frequent_value_and_index(tmp_answers)
                    new_context = all_context[max_index].copy()
                    for context_index in range(len(all_context)):
                        all_context[context_index] = new_context.copy()
                    extracted_answer = most_frequent_ans
                else:
                    sess = utils.get_function(model_name)(set_path, system_prompt, user_prompt, model_name, i, sess, bool_list, 0)
                    extracted_answer = utils.get_answer(set_path, model_name, i, bool_list, 0).lower()


                if answer.lower().replace(" ", "") in extracted_answer.lower().replace(" ", ""):
                    accs.append(1)
                    break
                else:
                    accs.append(0)
            if len(accs) < 3:
                accs += [1]*(3-len(accs))

            all_accs.append(accs)
            acc_saved_folder = utils.get_result_folder(model_name, bool_list, language)
            with open(os.path.join(acc_saved_folder, "Accuracy_for_all.json"), "w") as file:
                file.write(json.dumps(all_accs))


        elif bool_list[0] == 0:
            if bool_list[1] == 8:
                with open("indexs.json", "r") as f:
                    indexs = json.load(f)
                hint_path = os.path.join("../data", str(indexs[int(tmp_set) - 1]), f"Hint_3.txt")
                with open(hint_path, "r") as file:
                    hint = file.read()
            elif bool_list[1] == 9:
                pass
            else:
                with open(os.path.join(set_path, f"Hint_3.txt"), "r") as file:
                    hint = file.read()


            if (bool_list[1] == 4) or (bool_list[1] == 9):
                user_prompt = f"""{asking_prompt}"""
            else:
                user_prompt = f"""{asking_prompt}

Hint: {hint}
"""

            if model_name != "InternVL2_5-78B-MPO-AWQ":
                tmp_answers = []
                for context_index in range(times):
                    tmp_context = []
                    _ = utils.get_function(model_name)(set_path, system_prompt, user_prompt, model_name, 2, tmp_context, bool_list, context_index, language)
                    tmp_answer = extracted_answer = utils.get_answer(set_path, model_name, 2, bool_list, context_index, language).lower()
                    tmp_answers.append(tmp_answer)
                most_frequent_ans, max_index = utils.get_most_frequent_value_and_index(tmp_answers)
                extracted_answer = most_frequent_ans
            else:
                _ = utils.get_function(model_name)(set_path, system_prompt, user_prompt, model_name, 2, sess,
                                                      bool_list, 0)
                extracted_answer = utils.get_answer(set_path, model_name, 2, bool_list, 0).lower()

            if answer.lower().replace(" ", "") in extracted_answer.lower().replace(" ", ""):
                all_accs.append(1)
            else:
                all_accs.append(0)
            acc_saved_folder = utils.get_result_folder(model_name, bool_list, language)
            with open(os.path.join(acc_saved_folder, "Accuracy_for_all.json"), "w") as file:
                file.write(json.dumps(all_accs))