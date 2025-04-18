
import re
from typing import List
import os
from metagpt.actions import Action
from metagpt.schema import Message
from qa_module import AsyncQA_tutorial, AsyncQA_Ori, AsyncQA_allrun
import config_path
import subprocess
import sys
import json
from pathlib import Path
from Statistics import global_statistics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class InputWriterAction(Action):
    PROMPT_TEMPLATE: str = """
    Your task is {requirement}.
    The similar foamfile is provided as follows:
    {tutorial_file}
    Please take this foamfile as a reference, which may help you to finish your task.
    According to your task, return ```your_code_here ``` with NO other texts,
    your code:
    """
    PROMPT_TEMPLATE_no_tutorial: str = """
    Your task is {requirement}.
    According to your task, return ```your_code_here ``` with NO other texts,
    your code:
    """
    PROMPT_Find: str = """
        Find the OpenFOAM foamfile that most closely matches the following foamfile:
        {file_name} in {file_folder} of case name: {case_name}
    """
    name: str = "InputWriterAction"

    PROMPT_TEMPLATE_allrun: str = """
        Your task is to write linux execution command allrun file to meet the user requirement: {requirement}.
        Note that you only need to focus on the requirements for the CFD simulation task without including any additional analysis or explanation (like postprocessing), as these additional analysis or explanations have already been taken into account in the previous input files. You only need to set up the command of main CFD task now (like generate grids, run preprocessing, and run XXfoam).
        The input file list is {file_list}.
        Here is a openfoam allrun file similar to the user requirements:
        {tutorial}
        Please take this file as a reference.
        The possible command list is
        {commands}
        In the command list, the following commands are **forbidden** and should **never** be used:
            - `setFields`
            - `changeDictionary`
        The possible run list is
        {runlists}
        Make sure the written linux execution command are coming from the above two lists.
        According to your task, return 
        ```bash
        your_allrun_file_here
        ``` 
        with **no other texts**. And replace the placeholder your_allrun_file_here with the actual Allrun file content. Do not return the placeholder, but instead return the actual file content.
        """
    PROMPT_TEMPLATE_allrun_rewrite: str = """
    Your task is {requirement}.
    The similar foamfile is provided as follows:
    {tutorial}
    Please take this foamfile as a reference, which may help you to finish your task.
    And please do not use changeDictionary command.
    According to your task, return 
    ```bash
    your_allrun_file_here
    ``` 
    with **no other texts**. And replace the placeholder your_allrun_file_here with the actual Allrun file content. Do not return the placeholder, but instead return the actual file content.
    """
    PROMPT_TEMPLATE_postprocessing_total: str = """
In the OpenFOAM simulation for '{CFD_task}', to post-process and extract '{dependent_var}', please analyze how to write the post-processing command in linux Allrun_postprocessing file and use a Python script to automatically analyze the '{dependent_var}' from the results generated in the postprocessing stage after the simulation runs.
note that the postprocessing function can only be selected from the following list:
{postprocessing_list}
Please first return the 'Allrun_postprocessing' file for linux execution as:
`Allrun_postprocessing` file begin ```
your_Allrun_postprocessing_here 
``` `Allrun_postprocessing` file end
with **no other texts**. And replace the placeholder your_Allrun_postprocessing_here with the actual Allrun_postprocessing file content. Do not return the placeholder, but instead return the actual file content.
And then return the corresponding Python script as:
Python script begin ```
your_python_code_here
``` Python script end
with **no other texts**. And replace the placeholder your_python_code_here with the actual generated Python script. Do not return the placeholder, but instead return the actual file content.
        """
    PROMPT_TEMPLATE_postprocessing_allrun: str = """
In the OpenFOAM simulation for '{CFD_task}', to post-process and extract '{dependent_var}', please first write the post-processing command to get the openfoam post-processing file, which could be used for a Python script to extract '{dependent_var}'
note that the postprocessing function can only be selected from the following list:
{postprocessing_list}
Please first return the 'Allrun_postprocessing' file for linux execution as:
`Allrun_postprocessing` file begin ```
your_Allrun_postprocessing_here
``` `Allrun_postprocessing` file end
with **no other texts**. And replace the placeholder your_Allrun_postprocessing_here with the actual Allrun_postprocessing file content. Do not return the placeholder, but instead return the actual file content.
        """
    PROMPT_TEMPLATE_postprocessing_allrun2: str = """
In the OpenFOAM simulation for '{CFD_task}', to post-process and extract '{dependent_var}', please first write the post-processing command to get the openfoam post-processing file, which could be used for a Python script to extract '{dependent_var}'
Note that the previous allrun for CFD task has already been executed, so you only need to provide the post-processing command. Do not include 'runApplication blockMesh' or any 'runApplication &Application' commands.
Additionally, do not include any if, echo, exit, or other error handling commands. 
The post-processing function can only be selected from the following list:
{postprocessing_list}
You can call the above post-processing function using either 'runApplication postProcess -func Specific_postprocessing_function' or '&Application -postProcess -func Specific_postprocessing_function'. 
Note that the former invokes postProcess for post-processing, while the latter uses the solver for post-processing. The choice between the two depends on the type of Specific_postprocessing_function.
Please source the tutorial run functions first by including this line:
. $WM_PROJECT_DIR/bin/tools/RunFunctions
Please first return the 'Allrun_postprocessing' file for linux execution as:
```Allrun_postprocessing
your_Allrun_postprocessing_here
```
with **no other texts**. And replace the placeholder your_Allrun_postprocessing_here with the actual Allrun_postprocessing file content. Do not return the placeholder, but instead return the actual file content.
        """
    PROMPT_TEMPLATE_postprocessing_allrun3: str = """
In the OpenFOAM simulation for '{CFD_task}', the CFD postprocessing task is '{CFD_postprocessing_task}'
please first write the post-processing command to get the openfoam post-processing file, which could be used for a Python script to complete the post-processing task.
And then transform the generated postprocessing file into VTK format by including this line:
foamToVTK -latestTime -fields '(Specific_postprocessing_file1 Specific_postprocessing_file2 ...)' ...
Note that the previous allrun for CFD task has already been executed, so you only need to provide the post-processing command. Do not include 'runApplication blockMesh' or any 'runApplication &Application' commands.
Additionally, do not include any if, echo, exit, or other error handling commands. 
The post-processing function can only be selected from the following list:
{postprocessing_list}
You can call the above post-processing function using either 'runApplication postProcess -func Specific_postprocessing_function' or '&Application -postProcess -func Specific_postprocessing_function'. 
Note that the former invokes postProcess for post-processing, while the latter uses the solver for post-processing. The choice between the two depends on the type of Specific_postprocessing_function.
Please source the tutorial run functions first by including this line:
. $WM_PROJECT_DIR/bin/tools/RunFunctions

Please return the 'Allrun_postprocessing' file for linux execution as:
```Allrun_postprocessing
your_Allrun_postprocessing_here
```
with **no other texts**. And replace the placeholder your_Allrun_postprocessing_here with the actual Allrun_postprocessing file content. Do not return the placeholder, but instead return the actual file content.
        
        """
    PROMPT_TEMPLATE_postprocessing_allrun_vtk: str = """
In the OpenFOAM simulation for '{CFD_task}', the CFD postprocessing task is '{CFD_postprocessing_task}'.
The required file list for post-processing is '{related_file_list}'.
please write the post-processing command to transform the generated postprocessing file into VTK format by including this line:
foamToVTK -latestTime -fields '(Specific_postprocessing_file1 Specific_postprocessing_file2 ...)' ...
Please source the tutorial run functions first by including this line:
. $WM_PROJECT_DIR/bin/tools/RunFunctions
Please return the 'Allrun_postprocessing' file for linux execution as:
```Allrun_postprocessing
your_Allrun_postprocessing_here
```
with **no other texts**. And replace the placeholder your_Allrun_postprocessing_here with the actual Allrun_postprocessing file content. Do not return the placeholder, but instead return the actual file content.
        
        """
    
    PROMPT_TEMPLATE_postprocessing_if_exist: str = """
In the OpenFOAM simulation for '{CFD_task}', the CFD postprocessing task is '{CFD_postprocessing_task}'.  
Before performing the postprocessing task, check whether the required files are present in the list '{file_list}'.  

For example:  
- If the postprocessing task involves `yPlus`, the list should contain the file `yPlus`.  
- If the postprocessing task involves velocity fields, the list should contain the file `U`.  

If the required file is present, return:  
{json_structure}
Otherwise, return:  
```No```
    """

    PROMPT_TEMPLATE_postprocessing_allrun_JSON: str = """
In the OpenFOAM simulation for '{CFD_task}', to post-process and extract '{dependent_var}', please first write the post-processing command to get the openfoam post-processing file, which could be used for a Python script to extract '{dependent_var}'
note that the postprocessing function can only be selected from the following list:
{postprocessing_list}
Please first return the 'Allrun_postprocessing' file for linux execution in the following JSON format:
```
{JSON_allrun_postprocessing}
```
with **no other texts**.
        """
        
    PROMPT_TEMPLATE_postprocessing_python: str = """
In the OpenFOAM simulation for '{CFD_task}', to post-process and extract '{dependent_var}', the following Linux command was executed:
```
{postprocessing_command}
```
This command generated a file at the following path: '{postprocessing_new_data_path}'.
Here are the first 50 lines of the file (if the file contains more than 50 lines):
```
{postprocessing_data}
```
Please write a Python script that reads this file, extracts '{dependent_var}', and saves it in the following JSON format as 'dependent_var.json' in the current directory of the Python script: 
```
{JSON_dependent_var}
```
Please return the corresponding Python script as:
```python
your_python_code_here
```
with **no other texts**. And replace the placeholder your_python_code_here with the actual generated Python script. Do not return the placeholder, but instead return the actual file content.
        """
    
    PROMPT_TEMPLATE_postprocessing_python_for_vtk: str = """

In the OpenFOAM simulation for '{CFD_task}', the CFD postprocessing task is '{CFD_postprocessing_task}'.  
The following Linux command was executed:  
```  
{postprocessing_command}  
```  
This command generated a VTK file at the following path: '{postprocessing_new_data_path}'.  
Please write a Python script that reads this file and completes the CFD postprocessing task: 
- If the CFD postprocessing task involves extracting a specific value, extract it and save it in the following JSON format as 'postprocessing_var.json' in the script's current directory:  
```  
{JSON_dependent_var}  
```  
- If the CFD postprocessing task involves plotting, generate the required plot and save it as a PNG file in the script's current directory.  

Please return the Python script in the following format:  
```python  
your_python_code_here  
```  
Replace the placeholder `your_python_code_here` with the actual Python script code. Do not include any other text, and provide only the complete Python script as the output.
        """
    PROMPT_TEMPLATE_python_env: str = """
For the following Python program:  
{python_text}  
Please return the required packages in the following format:
Python env list begin ```
Your_python_env_list_here
``` Python env list end
with **no other texts**.
        """
    PROMPT_TEMPLATE_postprocessing_rewrite: str = """
When a Python program encounters an error during execution:  
{error}  
Here is the Python script being executed, 
{python_text}  

Please determine whether this error is caused by the Python environment or the content of the program.  
If the error is due to the Python environment, return a Linux command using Python subprocess to install or update the required library version as:  
###  
python_linux_command  
###  
with **no other texts**. 

If the error is due to the program content, return the corrected Python script as:  
```
Python_code_here  
```  
with **no other texts**.  
        """
    async def run(self, with_messages:List[Message]=None, **kwargs) -> Message:

        file_list = []

        async_qa_tutorial = AsyncQA_tutorial()
        async_qa = AsyncQA_Ori()
        document_text = self.read_openfoam_tutorials(f"{config_path.Database_PATH}/openfoam_tutorials.txt")
        allrun_file_path = f'{config_path.Case_PATH}/Allrun'
        postprocessing_python_path = f'{config_path.Case_PATH}/postprocessing_python.py'
        postprocessing_allrun_path = f'{config_path.Case_PATH}/Allrun_postprocessing'
        cfd_task = with_messages[0]
        error_log_path = f"{config_path.Case_PATH}/error_log.json"
        if global_statistics.Run_loop == 0:
            # delete error_log
            if os.path.exists(error_log_path):
                os.remove(error_log_path)

        postprocessing_var_path = Path(config_path.Case_PATH) / "postprocessing_var.json"
        allrun_outfile_path = Path(config_path.Case_PATH) / "Allrun.out"
        allrun_postprocessingout_path = Path(config_path.Case_PATH) / "Allrun_postprocessing.out"
        if os.path.exists(postprocessing_var_path) and os.path.isfile(postprocessing_var_path):
            if global_statistics.Executability == 0:
                global_statistics.Executability = 6
        # elif os.path.exists(allrun_postprocessingout_path) and os.path.isfile(allrun_postprocessingout_path):
        #     if global_statistics.Executability == 0:
        #         global_statistics.Executability = 4
        # elif os.path.exists(allrun_outfile_path) and os.path.isfile(allrun_outfile_path):
        #     if global_statistics.Executability == 0:
        #         global_statistics.Executability = 3

        similarity_matrix = self.calculate_similarity(error_log_path)
        if similarity_matrix is not None:
            print('1-2','1-3','2-3')
            print(similarity_matrix[0,1],similarity_matrix[0,2],similarity_matrix[1,2])
            if similarity_matrix[0,2] > 0.9 or similarity_matrix[1,2] > 0.9:
                if config_path.temperature < 0.5:
                    config_path.temperature = 0.5

                elif config_path.temperature == 0.5:
                    if config_path.If_all_files:
                        config_path.If_all_files = False
                    elif config_path.If_RAG:
                        config_path.If_RAG = False
            else:
                config_path.temperature = 0.01
        print('temperature:',config_path.temperature)
        print('If_RAG:',config_path.If_RAG)
        print('If_all_files:',config_path.If_all_files)
        for i in with_messages[1:]:
            
            if global_statistics.Executability < 3:

                # need to judge whether to write/rewrite allrun_file
                # first wirte allrun: need to do after file generatation
                file_name = self.parse_flie_name(i.content)
                file_list.append(file_name)
                
                IF_rewrite = self.parse_rewirte(i.content)
                # need to judge whether to write/rewrite allrun_file
                
                if 'Allrun' in file_name:
                    
                    allrun_write = "None"

                    if os.path.exists(allrun_file_path) and 'rewrite' not in IF_rewrite:

                        print(f"Allrun file already exists. Skipping...")

                        with open(allrun_file_path, 'r', encoding='utf-8') as allrun_file:

                            allrun_write = allrun_file.read()
                            print('allrun_write2:',allrun_write)

                    elif os.path.exists(allrun_file_path) and 'rewrite' in IF_rewrite:

                        print(f"Allrun file is going to be rewritten...")
                        file_list = self.read_files(config_path.Case_PATH)

                        find_tutorial = self.read_tutorial()
                        #print("find_tutorial:",find_tutorial)
                        case_name = self.get_case_name(find_tutorial)
                        #print("case_name:",case_name)
                        allrun_tutorial = self.get_allrun_tutorial(case_name)

                        promt_allrun_rewrite = self.PROMPT_TEMPLATE_allrun_rewrite.format(requirement=i.content, tutorial = allrun_tutorial)
                        rsp = await async_qa.ask(promt_allrun_rewrite)
                        code_text = self.parse_allrun_new(rsp)
                        print('rewritten allrun file:',code_text)
                        self.save_file(allrun_file_path, code_context=str(code_text))
                        

                else: # not allrun file

                    folder_name = self.parse_folder_name(i.content)
                    IF_rewrite = self.parse_rewirte(i.content)

                    file_path = f"{config_path.Case_PATH}/{folder_name}/{file_name}"
                    case_name_true = os.path.basename(config_path.Case_PATH)
                    
                    if os.path.exists(file_path) and 'rewrite' not in IF_rewrite:
                        print(f"File {file_name} already exists in {folder_name}. Skipping...")
                        continue
                    
                    if config_path.If_RAG:
                        if config_path.tasks >= 3:
                            case_info = self.read_similar_case(f"{config_path.Para_PATH}/find_tutorial.txt")
                        else:
                            case_info = self.read_similar_case(f"{config_path.Case_PATH}/find_tutorial.txt")
                        print('case_info:',case_info)
                        case_name = case_info['case_name']
                        case_domain = case_info['case_domain']
                        case_category = case_info['case_category']
                        case_solver = case_info['case_solver']
                        similar_file = f"```input_file_begin: input {file_name} file of case {case_name} (domain: {case_domain}, category: {case_category}, solver:{case_solver})"
                        
                        tutorial_file = self.find_similar_file(similar_file,document_text)
                        print("tutorial_file:",tutorial_file)
                        if tutorial_file == "None":
                            prompt_find = self.PROMPT_Find.format(file_name=file_name, file_folder=folder_name, case_name = case_name_true)
                            rsp = await async_qa_tutorial.ask(prompt_find)
                            result = rsp["result"]
                            print("find_similar_foamfile:", result)
                            doc = rsp["source_documents"]
                            tutorial_file = doc[0].page_content
                            print("find_tutorial_file:",tutorial_file)

                        print(f"File {file_name} is going to be written")
                        
                        prompt = self.PROMPT_TEMPLATE.format(requirement=i.content, tutorial_file = tutorial_file)
                    else:
                        prompt = self.PROMPT_TEMPLATE_no_tutorial.format(requirement=i.content)
                        
                    rsp = await async_qa.ask(prompt)
                    code_text = self.parse_context(rsp)
                    print('folder_name',folder_name)
                    print('file_name',file_name)
                    if folder_name.strip() and file_name.strip():
                        self.save_file(file_path, code_context=str(code_text))
                    else:
                        print("Folder name or file name is empty, skipping save operation.")
                        
            elif global_statistics.Executability == 3 and os.path.exists(postprocessing_allrun_path) and config_path.tasks>=2:
                print('rewrite for Allrun_postprocessing')
                error_log_path = f"{config_path.Case_PATH}/error_postprocessing_log.json"
                if global_statistics.Postprocess_loop == 0:
                    # delete error_log
                    if os.path.exists(error_log_path):
                        os.remove(error_log_path)

                similarity_matrix = self.calculate_similarity(error_log_path)
                if similarity_matrix is not None:
                    print('1-2','1-3','2-3')
                    print(similarity_matrix[0,1],similarity_matrix[0,2],similarity_matrix[1,2])
                    if similarity_matrix[0,2] > 0.9 or similarity_matrix[1,2] > 0.9:
                        config_path.temperature = 0.5
                    else:
                        config_path.temperature = 0.01
                
                promt_allrun_postprocessing_rewrite = i.content
                if promt_allrun_postprocessing_rewrite != "None":
                    rsp = await async_qa.ask(promt_allrun_postprocessing_rewrite)
                    print('Executability = 3, to rewrite postprocessing:',rsp)
                    
                    Allrun_postprocessing = self.parse_post_processing_new(rsp)
                    print('Allrun_postprocessing:',Allrun_postprocessing)
                    Allrun_postprocessing_path = f'{config_path.Case_PATH}/Allrun_postprocessing'

                    self.save_file(Allrun_postprocessing_path, code_context=str(Allrun_postprocessing))
                
            elif global_statistics.Executability == 3 and not os.path.exists(postprocessing_allrun_path) and config_path.tasks>=2:

                config_path.temperature = 0.0
                print('temperature:',config_path.temperature)
                CFD_task, CFD_post_tasks = self.load_CFD_tasks(f"{config_path.Case_PATH}/CFD_tasks.json")
                #controlDict_text = self.read_controlDict(config_path.Case_PATH)
                postprocessing_list = self.read_postprocess_commands(config_path.Database_PATH)
                #prompt_postprocessing = self.PROMPT_TEMPLATE_postprocessing.format(CFD_task=CFD_task, dependent_var=dependent_vars[0], controlDict_text=controlDict_text, postprocessing_list = postprocessing_list)
                #prompt_postprocessing = self.PROMPT_TEMPLATE_postprocessing_allrun_JSON.format(CFD_task=CFD_task, dependent_var=dependent_vars[0], postprocessing_list = postprocessing_list, JSON_allrun_postprocessing = JSON_allrun_postprocessing)
                end_time = self.get_end_time(config_path.Case_PATH)
                print('endTime',end_time)
                
                end_Time_file_list = self.get_files_in_endTime(end_time)

                json_structure = """
                {
                    "file_names": ["specific_file_name1", "specific_file_name2", ...]
                }
                """
                prompt_postprocessing_if_exist = self.PROMPT_TEMPLATE_postprocessing_if_exist.format(CFD_task=CFD_task, CFD_postprocessing_task = CFD_post_tasks, file_list = end_Time_file_list,json_structure = json_structure)
                rsp = await async_qa.ask(prompt_postprocessing_if_exist)
                if 'No' in rsp:
                    prompt_postprocessing = self.PROMPT_TEMPLATE_postprocessing_allrun3.format(CFD_task=CFD_task, CFD_postprocessing_task = CFD_post_tasks, postprocessing_list = postprocessing_list)
                else:
                    json_match = re.search(r'```json(.*)```', rsp, re.DOTALL)
                    json_str = json_match.group(1) if json_match else None
                    if json_str:
                        related_file_list = self.parse_json_output(json_str)
                        print("related_file_list:", related_file_list)
                    else:
                        print("No JSON found in the provided text.")
                        sys.exit()

                    prompt_postprocessing = self.PROMPT_TEMPLATE_postprocessing_allrun_vtk.format(CFD_task=CFD_task, CFD_postprocessing_task = CFD_post_tasks, related_file_list = related_file_list)
                
                #prompt_postprocessing = self.PROMPT_TEMPLATE_postprocessing_allrun2.format(CFD_task=CFD_task, dependent_var=dependent_vars[0], postprocessing_list = postprocessing_list)
                print('prompt_postprocessing:',prompt_postprocessing)

                rsp = await async_qa.ask(prompt_postprocessing)
                print('postprocessing_rsp:',rsp)

                Allrun_postprocessing = self.parse_post_processing_new(rsp)
                #print('Allrun_postprocessing:',Allrun_postprocessing)

                Allrun_postprocessing_path = f'{config_path.Case_PATH}/Allrun_postprocessing'

                self.save_file(Allrun_postprocessing_path, code_context=str(Allrun_postprocessing))

            elif (global_statistics.Executability == 4 or global_statistics.Executability == 5) and os.path.exists(postprocessing_python_path) and config_path.tasks>=2:
                
                print('rewrite for Allrun_postprocessing and postprocessing_python')

                error_log_path = f"{config_path.Case_PATH}/postprocessing_python_error.json"
                if global_statistics.Postprocess_loop == 0:
                    # delete error_log
                    if os.path.exists(error_log_path):
                        os.remove(error_log_path)

                similarity_matrix = self.calculate_similarity(error_log_path)
                if similarity_matrix is not None:
                    print('1-2','1-3','2-3')
                    print(similarity_matrix[0,1],similarity_matrix[0,2],similarity_matrix[1,2])
                    if similarity_matrix[0,2] > 0.9 or similarity_matrix[1,2] > 0.9:
                        config_path.temperature = 0.5
                    else:
                        config_path.temperature = 0.01
                promt_postprocessing_rewrite = i.content
                rsp = await async_qa.ask(promt_postprocessing_rewrite)
                
                print('Executability = 4/5, to rewrite postprocessing:',rsp)

                # save new python file

                python_script = self.parse_python_new(rsp)

                self.save_file(postprocessing_python_path, code_context=str(python_script))
                # Allrun_postprocessing = self.parse_Modified_post_processing(rsp)
                # print('Allrun_postprocessing:',Allrun_postprocessing)
                # Allrun_postprocessing_path = f'{config_path.Case_PATH}/Allrun_postprocessing'

                # self.save_file(Allrun_postprocessing_path, code_context=str(Allrun_postprocessing))

                # postprocessing_python = self.parse_Modified_python(rsp)
                
                # print('postprocessing_python:',postprocessing_python)
                # self.save_file(postprocessing_python_path, code_context=str(postprocessing_python))
                # prompt_python_env = self.PROMPT_TEMPLATE_python_env.format(python_text = postprocessing_python)
                # print('prompt_python_env:',prompt_python_env)
                # rsp = await async_qa.ask(prompt_python_env) 
                # print('rsp_python_env:',rsp)
                # python_env = self.parse_python_env(rsp)
                # print('parse_python_env:',python_env)
                # libraries = re.findall(r'\b\w+\b', python_env.strip())
                # print('python_env:',libraries)
                # # Check for each package and install if necessary
                # for package in libraries:
                #     try:
                #         __import__(package)
                #         print(f"{package} is already installed.")
                #     except ImportError:
                #         print(f"{package} is not installed. Installing...")
                #         self.install_package(package)

                # print("All required packages are installed.")
                
                
                #global_statistics.Executability = 3
                
            elif (global_statistics.Executability == 4 or global_statistics.Executability == 5) and not os.path.exists(postprocessing_python_path) and config_path.tasks>=2:
                config_path.temperature = 0.01
                print('first generate python script')
                CFD_task, CFD_post_tasks = self.load_CFD_tasks(f"{config_path.Case_PATH}/CFD_tasks.json")
                #controlDict_text = self.read_controlDict(config_path.Case_PATH)
                postprocessing_data_path = f"{config_path.Case_PATH}/new_files_data.json"
                all_new_files, new_file_paths = self.load_new_files_data(postprocessing_data_path)
                postprocessing_list = self.read_postprocess_commands(config_path.Database_PATH)
                
                JSON_dependent_var = """ 
                    {
                        "postprocessing_var": "specific_value"
                    }
                """
                print("JSON_dependent_var:",JSON_dependent_var)
                print('new_file_paths:',new_file_paths)
                print('all_new_files:',all_new_files)
                # 目前只考虑单个new_file_paths，即后处理只增加单个文件
                #postprocessing_data = self.read_first_50_lines(new_file_paths[0])

                postprocessing_command = self.read_postprocessing_command()
                prompt_postprocessing = self.PROMPT_TEMPLATE_postprocessing_python_for_vtk.format(CFD_task=CFD_task, 
                                                                          CFD_postprocessing_task = CFD_post_tasks,
                                                                          postprocessing_command = postprocessing_command,
                                                                          postprocessing_new_data_path = new_file_paths,
                                                                          JSON_dependent_var = JSON_dependent_var)

                #prompt_postprocessing = self.PROMPT_TEMPLATE_postprocessing.format(CFD_task=CFD_task, dependent_var=dependent_vars[0], controlDict_text=controlDict_text, postprocessing_list = postprocessing_list)
                # prompt_postprocessing = self.PROMPT_TEMPLATE_postprocessing_python.format(CFD_task=CFD_task, 
                #                                                                           postprocessing_command = postprocessing_command, 
                #                                                                           postprocessing_new_data_path = new_file_paths[0], 
                #                                                                           postprocessing_data = postprocessing_data, 
                #                                                                           JSON_dependent_var = JSON_dependent_var)
                
                print('prompt_postprocessing_python:',prompt_postprocessing)
                rsp = await async_qa.ask(prompt_postprocessing) 
                print('postprocessing_rsp:',rsp)
                posrprocessing_python_script = self.parse_python_new(rsp)
                print('posrprocessing_python_script:',posrprocessing_python_script)
                self.save_file(postprocessing_python_path, code_context=str(posrprocessing_python_script))
            elif global_statistics.Executability == 6 and config_path.tasks>=2:
                print('specific_case already run,EXE = ',global_statistics.Executability)
                
            else:
                print('wrong Executability!!')
                sys.exit()
                

        #first write allrun        
        if not os.path.exists(allrun_file_path):
            #write allrun
            requirement = cfd_task.content # need to be fixed

            async_qa_allrun = AsyncQA_allrun()
            runlists = ['isTest', 'getNumberOfProcessors','getApplication','runApplication','runParallel','compileApplication','cloneCase','cloneMesh']
            commands = self.read_commands(config_path.Database_PATH)
            file_list = self.read_files(config_path.Case_PATH)

            find_tutorial = self.read_tutorial()
            #print("find_tutorial:",find_tutorial)
            case_name = self.get_case_name(find_tutorial)
            #print("case_name:",case_name)
            allrun_tutorial = self.get_allrun_tutorial(case_name)
            #print("allrun_tutorial:",allrun_tutorial)

            prompt_allrun = self.PROMPT_TEMPLATE_allrun.format(
                requirement=requirement, 
                tutorial = allrun_tutorial,
                file_list = file_list, 
                commands = commands, 
                runlists = runlists)
            
            rsp = await async_qa_allrun.ask(prompt_allrun) 
            result = rsp["result"]
            #doc = rsp["source_documents"]
            #print("allrun_source_documents:",doc[0])
            #print("allrun:",result)
            allrun_write = self.parse_allrun_new(result)
            with open(allrun_file_path, 'w') as outfile:  
                outfile.write(allrun_write)

            print('allrun_write:',allrun_write)
        #first write postprocessing in controlDict
        # 在OpenFOAM的{case_name}模拟中，想要后处理提取{dependent_var}，请分析如何在controlDict中写后处理程序并在运行后得到的postprocessing中利用python程序自动分析得到{dependent_var}
        # v2: 在controlDict中会遇到在run的阶段将后处理改没的情况，所以还是写在allrun中吧

        # if not os.path.exists(postprocessing_allrun_path) and global_statistics.Executability == 3:
        #     CFD_task, independent_vars, dependent_vars, samples, Specific_CFD_tasks, Multi_CFD_tasks = self.load_parameters(config_path.Para_PATH)
        #     #controlDict_text = self.read_controlDict(config_path.Case_PATH)
        #     postprocessing_list = self.read_postprocess_commands(config_path.Database_PATH)
        #     #prompt_postprocessing = self.PROMPT_TEMPLATE_postprocessing.format(CFD_task=CFD_task, dependent_var=dependent_vars[0], controlDict_text=controlDict_text, postprocessing_list = postprocessing_list)
        #     prompt_postprocessing = self.PROMPT_TEMPLATE_postprocessing_allrun.format(CFD_task=CFD_task, dependent_var=dependent_vars[0], postprocessing_list = postprocessing_list)
        #     print('prompt_postprocessing:',prompt_postprocessing)
        #     rsp = await async_qa.ask(prompt_postprocessing)
        #     print('postprocessing_rsp:',rsp)

        #     Allrun_postprocessing = self.parse_post_processing(rsp)
        #     print('Allrun_postprocessing:',Allrun_postprocessing)

        #     Allrun_postprocessing_path = f'{config_path.Case_PATH}/Allrun_postprocessing'

        #     self.save_file(Allrun_postprocessing_path, code_context=str(Allrun_postprocessing))

            
        # if not os.path.exists(postprocessing_python_path) and global_statistics.Executability == 4:
            
        #     CFD_task, independent_vars, dependent_vars, samples, Specific_CFD_tasks, Multi_CFD_tasks = self.load_parameters(config_path.Para_PATH)
        #     #controlDict_text = self.read_controlDict(config_path.Case_PATH)
        #     postprocessing_data_path = f"{config_path.Case_PATH}/new_files_data.json"
        #     all_new_files, new_file_paths = self.load_new_files_data(postprocessing_data_path)
        #     postprocessing_list = self.read_postprocess_commands(config_path.Database_PATH)
        #     dependent_var=dependent_vars[0]
        #     JSON_dependent_var = f"""
        #         {
        #             {dependent_var}: specific_value,
        #         }
        #     """
            
        #     postprocessing_data = self.read_first_30_lines(postprocessing_data_path)
            
        #     #prompt_postprocessing = self.PROMPT_TEMPLATE_postprocessing.format(CFD_task=CFD_task, dependent_var=dependent_vars[0], controlDict_text=controlDict_text, postprocessing_list = postprocessing_list)
        #     prompt_postprocessing = self.PROMPT_TEMPLATE_postprocessing_python.format(CFD_task=CFD_task, dependent_var=dependent_vars[0], postprocessing_data_path = postprocessing_data_path, postprocessing_data = postprocessing_data, JSON_dependent_var = JSON_dependent_var)
            
        #     print('prompt_postprocessing:',prompt_postprocessing)
        #     rsp = await async_qa.ask(prompt_postprocessing) 
        #     print('postprocessing_rsp:',rsp)
        #     sys.exit()
            
            
        #     Allrun_postprocessing = self.parse_post_processing(rsp)
        #     print('Allrun_postprocessing:',Allrun_postprocessing)
        #     Allrun_postprocessing_path = f'{config_path.Case_PATH}/Allrun_postprocessing'

        #     self.save_file(Allrun_postprocessing_path, code_context=str(Allrun_postprocessing))

        #     postprocessing_python = self.parse_python(rsp)
            
        #     print('postprocessing_python:',postprocessing_python)
        #     self.save_file(postprocessing_python_path, code_context=str(postprocessing_python))
        #     prompt_python_env = self.PROMPT_TEMPLATE_python_env.format(python_text = postprocessing_python)
        #     print('prompt_python_env:',prompt_python_env)
        #     rsp = await async_qa.ask(prompt_python_env) 
        #     print('rsp_python_env:',rsp)
        #     python_env = self.parse_python_env(rsp)
        #     print('parse_python_env:',python_env)
        #     libraries = re.findall(r'\b\w+\b', python_env.strip())
        #     print('python_env:',libraries)
        #     # Check for each package and install if necessary
        #     for package in libraries:
        #         try:
        #             __import__(package)
        #             print(f"{package} is already installed.")
        #         except ImportError:
        #             print(f"{package} is not installed. Installing...")
        #             self.install_package(package)

        #     print("All required packages are installed.")

        return "dummpy message"

    
    @staticmethod
    def parse_flie_name(rsp):
        pattern = r"a OpenFoam (.*) foamfile"
        match = re.search(pattern, rsp, re.DOTALL)
        your_task_flie = match.group(1) if match else ''
        return your_task_flie

    @staticmethod
    def parse_folder_name(rsp):
        pattern = r"foamfile in (.*) folder that"
        match = re.search(pattern, rsp, re.DOTALL)
        your_task_folder = match.group(1) if match else ''
        return your_task_folder
      
    @staticmethod
    def parse_context(rsp):
        pattern = r"(FoamFile.*?)(?:```|$)"
        match = re.search(pattern, rsp, re.DOTALL)
        if match:
            your_task_flie = match.group(1) 
        else:
            match2 = re.search(r'```(?:.*?\n)(.*?)\n```', rsp, re.DOTALL)
            if match2:
                your_task_flie =  match2.group(1) 
            else:
                your_task_flie = rsp
        return your_task_flie
    
    @staticmethod
    def parse_rewirte(rsp):
        pattern = r"to (.*) a OpenFoam"
        match = re.search(pattern, rsp, re.DOTALL)
        your_task_flie = match.group(1) if match else ''
        return your_task_flie
    def find_similar_file(self, start_string, document_text):

        start_pos = document_text.find(start_string)
        if start_pos == -1:
            return "None"
        
        end_pos = document_text.find("input_file_end.", start_pos)
        if end_pos == -1:
            return "None"
        
        return document_text[start_pos:end_pos + len("input_file_end.")]
    def read_similar_case(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                # 初始化要读取的字段
                case_info = {
                    'case_name': None,
                    'case_domain': None,
                    'case_category': None,
                    'case_solver': None
                }
                
                for line in file:
                    line = line.strip()
                    if line.startswith('case name:'):
                        case_info['case_name'] = line.split('case name:')[1].strip()
                    elif line.startswith('case domain:'):
                        case_info['case_domain'] = line.split('case domain:')[1].strip()
                    elif line.startswith('case category:'):
                        case_info['case_category'] = line.split('case category:')[1].strip()
                    elif line.startswith('case solver:'):
                        case_info['case_solver'] = line.split('case solver:')[1].strip()

                return case_info
            
        except FileNotFoundError:
            return f"file {file_path} not found"
    
    # def read_similar_case(self, file_path):
    #     try:
    #         with open(file_path, 'r', encoding='utf-8') as file:
    #             # 读取文件的第一行
    #             first_line = file.readline().strip()
                
    #             case_name_pos = first_line.find('case name:')
    #             if case_name_pos == -1:
    #                 return "None"
                
    #             case_name = first_line[case_name_pos + len('case name:'):].strip()
    #             return case_name
        
    #     except FileNotFoundError:
    #         return f"file {file_path} not found"
    def read_openfoam_tutorials(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:

                content = file.read()
                return content
        except FileNotFoundError:
            return f"file {file_path} not found"
        except Exception as e:
            return f"reading file meet error: {e}"
    def save_file(self, file_path: str, code_context: str) -> None:

        directory = os.path.dirname(file_path)
        # Create the folder path if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        with open(file_path, 'w') as file:
            file.write(code_context)  # 将代码写入文件

        print(f"File saved successfully at {file_path}")
        

    
        
    def read_file_content(self, file_path):
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            return None
    def read_commands(self, database_path):

        file_path = f"{database_path}/openfoam_commands.txt"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        with open(file_path, 'r') as file:
            commands = [line.strip() for line in file if line.strip()]
    
        return commands
    def read_files(self, base_path):
        file_names = []   
        base_depth = base_path.rstrip(os.sep).count(os.sep) 
        for root, dirs, files in os.walk(base_path):
            current_depth = root.rstrip(os.sep).count(os.sep)
            if current_depth == base_depth + 1: 
                for file in files:
                    file_path = os.path.join(root, file)  

                    try:
                        with open(file_path, 'r') as file_handle:
                            content = file_handle.read() 
                            file_names.append(file)
                    except UnicodeDecodeError:
                        print(f"Skipping file due to encoding error: {file_path}")
                        continue
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")
        return file_names
    def read_tutorial(self):
        if config_path.tasks>=3:
            save_path = config_path.Para_PATH
        else:
            save_path = config_path.Case_PATH
        file_path = f"{save_path}/find_tutorial.txt"
        with open(file_path, 'r') as file_handle:
            content = file_handle.read() 
        return content
    def get_case_name(self, content):
        match = re.search(r'case name:\s*(.+)', content)
        your_task_folder = match.group(1).strip() if match else 'None'
        return your_task_folder
    
    def get_allrun_tutorial(self,case_name):

        filename = 'openfoam_allrun.txt' 
        file_path = f"{config_path.Database_PATH}/{filename}"
        end_marker = 'input_file_end.```'  
        with open(file_path, 'r') as file:  
            lines = file.readlines()  
        extracted_content = []
        found_keyword = False  
        for line in lines:  
            if found_keyword:  
                if end_marker in line:  
                    break  
                extracted_content.append(line)  
            elif case_name in line:  
                found_keyword = True  
                continue  

        return ''.join(extracted_content)  
    

    def parse_allrun(self, allrun_total):
        print('allrun_total:',allrun_total)

        match = re.search(r'```(?:.*?\n)(.*?)\n```', allrun_total, re.DOTALL)
        allrun_text = match.group(1).strip() if match else 'None'
        return allrun_text
    def install_package(self,package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
    def load_parameters(self, file_path):
        with open(Path(file_path) / "Parameter.txt", "r") as file:
            parameters = json.load(file)
        
        CFD_task = parameters["CFD_task"]
        independent_vars = parameters["independent_vars"]
        dependent_vars = parameters["dependent_vars"]
        samples = parameters["samples"]
        Specific_CFD_tasks = parameters["Specific_CFD_tasks"]
        Multi_CFD_tasks = parameters["Multi_CFD_tasks"]
        
        return CFD_task, independent_vars, dependent_vars, samples, Specific_CFD_tasks, Multi_CFD_tasks
    
    def load_CFD_tasks(self, file_path):
        """
        Loads JSON data from a specified file path.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            dict: The loaded JSON data as a dictionary, or None if an error occurs.
        """
        try:
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
            print(f"JSON data successfully loaded from {file_path}")
            CFD_task = data["CFD_simulation_task"]
            CFD_post_task = data["CFD_postprocessing_task"]
            return CFD_task, CFD_post_task
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def read_controlDict(self, file_path):
        
        controlDict_path = Path(file_path) / "system" / "controlDict"
        if controlDict_path.exists():
            with open(controlDict_path, "r") as file:
                controlDict_text = file.read()
            return controlDict_text
        else:
            print("controlDict file not found.")
            sys.exit()
            return None

    def read_postprocess_commands(self, file_path):
        try:
            with open(Path(file_path) / "postprocessing_commands.txt", "r") as file:
                command_list = file.read().splitlines()
            return command_list
        except FileNotFoundError:
            print(f"File {file_path}/postprocessing_commands.txt not found.")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
            
    def calculate_similarity(self, json_file_path):
        # 检查文件是否存在且包含足够的数据
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as file:
                error_log = json.load(file)
                # 读取最新3次的迭代内容
                latest_errors = list(error_log.values())[-3:]
                if len(latest_errors) < 3:
                    print("Not enough data for similarity calculation.")
                    return None
                # 使用TF-IDF向量化文本
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(latest_errors)
                # 计算相似性
                cos_sim = cosine_similarity(tfidf_matrix)
                return cos_sim
        else:
            print("Error log file does not exist.")
            return None
        
    def load_new_files_data(self, json_file_path):
        """
        读取 JSON 文件并返回 all_new_files 和 new_file_paths 列表.
        
        参数:
            json_file_path (str): JSON 文件路径.
            
        返回:
            tuple: 包含 all_new_files 和 new_file_paths 的元组.
        """
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        print("Loaded JSON data:", data)

        all_new_files = data.get("all_new_files", [])
        new_file_paths = data.get("new_end_file_paths", []) 
        return all_new_files, new_file_paths
    
    def read_first_50_lines(self, file_path):
        # Read the file and keep only the first 30 lines as a single string
        with open(file_path, "r") as file:
            lines = file.readlines()
        
        # Join the first 30 lines into a single string
        first_50_lines = "".join(lines[:50])
        return first_50_lines
    
    def read_postprocessing_command(self):
        """读取 Allrun_postprocessing 文件内容"""
        file_path = f"{config_path.Case_PATH}/Allrun_postprocessing"
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            return f"Error: The file at {file_path} was not found."
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"
        
    def get_end_time(self, address):
        control_dict_path = os.path.join(address, 'system', 'controlDict')
        if not os.path.isfile(control_dict_path):
            print("controlDict file not found.")
            return
        
        endtime_value = None

        with open(control_dict_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.strip().startswith('endTime'):
                    endtime_value = line.split('endTime', 1)[1].strip().strip(';').replace(" ", "")
                    break

        if endtime_value is None:
            print("endTime not found in controlDict.")
            return
        
        return endtime_value
    
    def get_files_in_endTime(self, end_time):
        # 构造文件夹路径
        folder_path = f"{config_path.Case_PATH}/{end_time}"
        
        # 检查文件夹是否存在
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            # 获取文件夹中所有文件（不包含子文件夹中的文件）
            file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            return file_list
        else:
            sys.exit()
            return [] 
    
    def parse_json_output(self, json_output):
        """
        Parse the given JSON string to extract the file names.
        
        Args:
            json_output (str): JSON string in the format:
                {
                    "file_names": ["specific_file_name1", "specific_file_name2", ...]
                }
        
        Returns:
            list: A list of file names if the JSON is valid and contains 'file_names'.
                Returns an empty list if the key does not exist or the JSON is invalid.
        """
        try:
            # Parse the JSON string
            data = json.loads(json_output)
            
            # Extract 'file_names' if present
            if "file_names" in data and isinstance(data["file_names"], list):
                return data["file_names"]
            else:
                return []  # Return empty list if 'file_names' key is missing or invalid
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return []  # Return empty list if JSON is invalid

    @staticmethod
    def parse_post_processing(rsp):
        pattern = r"`Allrun_postprocessing` file begin ```(.*)``` `Allrun_postprocessing` file end"
        match = re.search(pattern, rsp, re.DOTALL)
        your_task_folder = match.group(1) if match else 'None'
        return your_task_folder
    @staticmethod
    def parse_post_processing_new(rsp):
        pattern = r"```Allrun_postprocessing(.*)```"
        match = re.search(pattern, rsp, re.DOTALL)
        your_task_folder = match.group(1) if match else 'None'
        return your_task_folder
    @staticmethod
    def parse_python(rsp):
        pattern = r"Python script begin ```(.*)``` Python script end"
        match = re.search(pattern, rsp, re.DOTALL)
        your_task_folder = match.group(1) if match else 'None'
        return your_task_folder
    @staticmethod
    def parse_python_new(rsp):
        pattern = r"```python(.*)```"
        match = re.search(pattern, rsp, re.DOTALL)
        your_task_folder = match.group(1) if match else 'None'
        return your_task_folder
    @staticmethod
    def parse_python_env(rsp):
        pattern = r"Python env list begin ```(.*)``` Python env list end"
        match = re.search(pattern, rsp, re.DOTALL)
        your_task_folder = match.group(1) if match else 'None'
        return your_task_folder
    
    @staticmethod
    def parse_Modified_post_processing(rsp):
        pattern = r"Modified `Allrun_postprocessing` file begin ```(.*)``` Modified `Allrun_postprocessing` file end"
        match = re.search(pattern, rsp, re.DOTALL)
        if match:
            your_task_folder = match.group(1) 
        else:
            pattern = r"```(.*)```"
            match = re.search(pattern, rsp, re.DOTALL)
            your_task_folder = match.group(1) if match else 'None'
        return your_task_folder
    @staticmethod
    def parse_Modified_python(rsp):
        pattern = r"Modified Python script begin ```(.*)``` Modified Python script end"
        match = re.search(pattern, rsp, re.DOTALL)
        your_task_folder = match.group(1) if match else 'None'
        return your_task_folder
    
    @staticmethod
    def parse_allrun_new(rsp):
        pattern = r"```bash(.*)```"
        match = re.search(pattern, rsp, re.DOTALL)
        your_task_folder = match.group(1) if match else 'None'
        return your_task_folder
        