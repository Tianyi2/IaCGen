## Code Files
1. [main.py](https://github.com/Tianyi2/IaCGen/blob/main/Code/main.py) contains the backbone of IaCGen.
2. [ablation_study.py](https://github.com/Tianyi2/IaCGen/blob/main/Code/ablation_study.py) contains the code used for ablation study where conversation history feature is disabled.
3. [security.py](https://github.com/Tianyi2/IaCGen/blob/main/Code/security.py) contains the code we used to perform security validation.
4. [user_intent.py](https://github.com/Tianyi2/IaCGen/blob/main/Code/user_intent.py) contains the code we used to perform user_intent validation.
5. [evalualtion/cloud_evaluation.py](https://github.com/Tianyi2/IaCGen/blob/main/Code/evaluation/cloud_evaluation.py) is the supporting code file which provide necessary functions for IaC template evaluation (yaml format, CloudFormation syntax, and live deployment).
6. [generation](https://github.com/Tianyi2/IaCGen/tree/main/Code/generation) folder contains the necessary support such as `prompts` and functions to interact with LLMs. (Currently the [cloud_generation.py](https://github.com/Tianyi2/IaCGen/blob/main/Code/generation/cloud_generation.py) is not used for IaCGen).




## Execute IaCGen
1. You can directly run the [main.py](https://github.com/Tianyi2/IaCGen/blob/main/Code/main.py) to execute the IaCGen in generating deployable IaC.
2. You can update the `llm_type` and `llm_type` variables to change the LLM.
3. You can edit the `start_row` and `end_row` to control which rows in the dataset you want to generate with IaCGen.
4. The default setting of the feedback mechanism 2 attempts for general, 4 attempts for detailed, 
and 4 attempts for human. To change the number of attempts for each type of feedback, you can edit the parameters of
IterativeTemplateGenerator class (`simple_level_max_iterations` for general feedback | `moderate_level_max_iterations` 
for detailed feedback | `advance_level_max_iterations` for human feedback). By giving a value of 0, you will turn off 
the type of feedback.
5. **Note**: To perform the evaluation of LLMs on IaC, you can first run the [main.py]() and run [security.py](https://github.com/Tianyi2/IaCGen/blob/main/Code/security.py) or [user_intent.py](https://github.com/Tianyi2/IaCGen/blob/main/Code/user_intent.py) if you want to validate security and user intent aspects.


## Supporting LLMs
Currently, models from the below providers are all supported by IaCGen. 
1. DeepSeek
2. OpenAI (GPT)
3. Claude
4. Gemini

You can update the `llm_model` to define the LLM model you want to use. The value of `llm_model` should follow the model name given by the LLM provider.

`llm_type` should be either value of `"gemini", "gpt", "claude", or "deepseek"`. When update the `llm_model`, you should update the `llm_type` to the corresponding provider name.



## Prompts
You can check the prompts in [prompt_for_cloud.py](https://github.com/Tianyi2/IaCGen/blob/main/Code/generation/prompts/prompt_for_cloud.py) and below.
### System Prompt
```
You are an expert in AWS CloudFormation template generation. Your task is to generate and improve templates based on feedback. Please write your complete CloudFormation YAML template inside <iac_template></iac_template> tags.
```

### Chain-of-Though User Input Prompt
```commandline
You are an expert AWS DevOps engineer with extensive experience in creating CloudFormation templates. Your task is to generate a valid, production ready and deployable AWS CloudFormation YAML template based on the following business need:

<business_need>
{{business_need_placeholder}}
</business_need>

Instructions:
1. Analyze the business need carefully.
2. Generate a complete CloudFormation YAML template that fulfills this need.
3. Be sure to fully specified all resources needed in the Resources section.
4. Ensure high accuracy and deployment success by following these guidelines:
   a. Start the template with 'AWSTemplateFormatVersion'.
   b. Include all necessary resources to meet the business need.
   c. Provide all required properties for each resource.
   d. Use proper YAML syntax and indentation.
   e. Follow AWS CloudFormation best practices and cloudformation-linter rules.
   f. End the template with the last property of the last resource.
5. Do not include any explanations, markdown formatting, or backticks in your output.

Before generating the final template, wrap your planning process in <template_planning> tags. In this section:
1. List the key AWS services mentioned or implied in the business need.
2. Outline the main sections of the CloudFormation template (e.g., Parameters, Mappings, Resources, Outputs).
3. Consider potential dependencies between resources and how to order them.
4. Think about any parameters or mappings that might be needed for flexibility.
5. Consider any outputs that would be useful for the user after stack creation.

This planning process will help reduce errors and improve deployment success rate. It's okay for this section to be quite long.

After your planning process, provide the complete CloudFormation YAML template as your final output.
```












