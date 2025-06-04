# IaCGen---First edition
`IaCGen` is a LLM improvement framework in Infrastructure-as-Code (IaC) generation.

`DPIaC-Eval` is the first deployablility-focused IaC benchmark that focuses on CloudFormation and AWS.


## Installation
1. Download the project
2. [Install AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) and [setup credentials](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-quickstart.html)
3. Download required libraries in the [requirement.txt](https://github.com/Tianyi2/IaCGen/blob/main/requirements.txt)
4. *Obtain the following LLM model inference API keys as appropriate. Currently IaCGen support all models from the following three providers:
- [OpenAI API](https://platform.openai.com/docs/quickstart/account-setup): for GPT-4o and o3-mini (Used in the paper)
- [Anthropic API](https://console.anthropic.com/): for Claude-3-5-Sonnet and Claude-3-7-Sonnet
- [DeepSeek API](https://platform.deepseek.com/): for DeepSeek-R1 and DeepSeek-S3
5. Add a `.env` file under the `IaCGen` directory with your own API key:
```
GEMIN_API_KEY=your_actual_gemini_api_key_here
CHATGPT_API_KEY=your_actual_gpt_api_key_here
CLAUDE_API_KEY=your_actual_claude_api_key_here
DEEPSEEK_API_KEY=your_actual_deepseek_api_key_here
```
6. Follow the instructions in [Code/README.md](https://github.com/Tianyi2/IaCGen/blob/main/Code/README.md) to execute the IaCGen.


## Project Structure
- You can check our `benchmark (DPIaC-Eval)` dataset under the [Data](https://github.com/Tianyi2/IaCGen/tree/main/Data) folder.
- You can check the `code for IaCGen` framework under the [Code](https://github.com/Tianyi2/IaCGen/tree/main/Code) folder. 
- **Note**: Please check the README.md file in each of the folders for a detailed description.
- **Note**: You can simply download the project and run the [main.py](https://github.com/Tianyi2/IaCGen/blob/main/Code/main.py) in [Code](https://github.com/Tianyi2/IaCGen/tree/main/Code) folder to test IaCGen. You can edit the variables in the last part of the Python file to control how you want to use IaCGen, such as the type of model and which IaC problem/s you want to test with. 


## Contribution
We welcome all forms of contribution! We aim to quantitatively and comprehensively evaluate the IaC code generation capabilities of large language models. If find bugs, issues, or have suggestions, please share them via GitHub Issues.  


## Acknowledgments
[IaC-Eval](https://github.com/autoiac-project/iac-eval)

