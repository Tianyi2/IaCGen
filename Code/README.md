## Code Files
1. main.py contains the backbone of IaCGen, while code from evaluation/cloud_evalution.py and generation/cloud_generation.py supports some features of IaCGen.
2. ablation_study.py contains the code used for ablation study where conversation history feature is disabled.
3. security.py contains the code we used to perform security validation.
4. user_intent.py contains the code we used to perform user_intent validation.

## Execute IaCGen
1. You can directly run the main.py to execute the IaCGen in generating deployable IaC.
2. You can update the `llm_type` and `llm_type` variables to change the LLM.
3. You can edit the `start_row` and `end_row` to control which rows in the dataset you want to generate with IaCGen.
4. The default setting of the feedback mechanism 2 attempts for general, 4 attempts for detailed, 
and 4 attempts for human. To change the number of attempts for each type of feedback, you can edit the parameters of
IterativeTemplateGenerator class (`simple_level_max_iterations` for general feedback | `moderate_level_max_iterations` 
for detailed feedback | `advance_level_max_iterations` for human feedback). By giving a value of 0, you will turn off 
the type of feedback.

## Note
To perform the evaluation of LLMs on IaC, you can first run the main.py and run security.py or user_intent.py if you want to validate security and user intent aspects.

