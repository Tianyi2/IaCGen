# Code
1. main.py contains the backbone of IaCGen, while code from evaluation/cloud_evalution.py and generation/cloud_generation.py supports some features of IaCGen.
2. ablation_study.py contains the code used for ablation study where conversation history feature is disabled.
3. security.py contains the code we used to perform security validation.
4. user_intent.py contains the code we used to perform user_intent validation.

To perform the evaluation of LLMs on IaC, you can first run the main.py and run security.py or user_intent.py if you want to validate security and user intent aspects.

