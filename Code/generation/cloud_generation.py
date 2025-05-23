import sys
import os
from openai import OpenAI
import anthropic

# Add the directory containing my_configs to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import google.generativeai as genai
from generation.prompts.prompt_for_cloud import TEMPLATE_GENERATE_PROMPT, SYSMTE_TEMPLATE_GENERATE_PROMPT, GPT_TEMPLATE_GENERATE_HELPER_PROMPT, TOP_PROMPT, BOTTOM_PROMPT
from my_configs.config import GEMIN_API_KEY, CHATGPT_API_KEY, CLAUDE_API_KEY, DEEPSEEK_API_KEY

OUTPUT_PATH = "llm_generated_data/template/"


def gemini_generate_cf_template(model, prompt, output_path=OUTPUT_PATH):  
    response = model.generate_content(
        prompt,
        generation_config = genai.GenerationConfig(
            max_output_tokens=4096,
            temperature=0.1,
        )
    )

    content = response.text

    # Remove planning session text
    position = content.find("</template_planning>")
    remove_length = len("</template_planning>")
    if position != -1:
        content = content[position+remove_length:]
    else:
        content = content  

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Generate a filename based on timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_path}template_{timestamp}.yaml"
    
    # Save the response to a file
    with open(filename, 'w') as f:
        f.write(content)
    
    return filename


def chatgpt_generate_cf_template(model, prompt, output_path=OUTPUT_PATH, llm_model="gpt-3.5-turbo"):
    """Generate content using OpenAI's GPT model."""
    response = model.chat.completions.create(
        model=llm_model,  # or another engine you are using
        max_tokens=4096,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "<template_planning>"
                    }
                ]
            }
        ]
    )

    # ToDo: It will not generate </template_planning>
    content = response.choices[0].message.content
    # Remove planning session text
    position = content.find("</template_planning>")
    remove_length = len("</template_planning>")
    if position != -1:
        content = content[position+remove_length:]
    else:
        content = content  
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Generate a filename based on timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_path}template_{timestamp}.yaml"
    
    # Save the response to a file
    with open(filename, 'w') as f:
        f.write(content)
    
    return filename


def claude_generate_cf_template(model, prompt, output_path=OUTPUT_PATH, llm_model="claude-3-5-sonnet-20241022"):
    """Generate content using Anthropic's Claude model."""
    response = model.messages.create(
        model=llm_model,
        max_tokens=4096,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "<template_planning>"
                    }
                ]
            }
        ]
    )
    content = response.content[0].text

    # Remove planning session text
    position = content.find("</template_planning>")
    remove_length = len("</template_planning>")
    if position != -1:
        content = content[position+remove_length:]
    else:
        content = content  
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Generate a filename based on timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_path}template_{timestamp}.yaml"
    
    # Save the response to a file
    with open(filename, 'w') as f:
        f.write(content)
    
    return filename


def deepseek_generate_cf_template(model, prompt, output_path=OUTPUT_PATH, llm_model="deepseek-reasoner"):
    response = model.chat.completions.create(
        model=llm_model,
        max_tokens=4096,
        temperature=0,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "<template_planning>", "prefix": True}
        ],
        stream=False
    )

    content = response.choices[0].message.content
    # ToDo: Remove planning session text
    # position = content.find("</template_planning>")
    # remove_length = len("</template_planning>")
    # if position != -1:
    #     content = content[position+remove_length:]
    # else:
    #     content = content  
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Generate a filename based on timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_path}template_{timestamp}.yaml"
    
    # Save the response to a file
    with open(filename, 'w') as f:
        f.write(content)
    
    return filename


def process_ioc_csv(input_csv, output_csv, model, llm_type, llm_model):
    """Process IOC CSV file and generate templates for each row
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        model: LLM model instance
        llm_type: Type of LLM being used (e.g., "gemini", "gpt", "claude")
        generate_template_fn: Function to generate template (e.g., gemini_generate_cf_template)
    """
    # Read the input CSV
    df = pd.read_csv(input_csv)
    
    # Process each row
    for index, row in df.iterrows():
        output_path = OUTPUT_PATH + llm_type + "/"
        
        try:
            if llm_type == "gemini":
                template_prompt = TOP_PROMPT + row["prompt"] + BOTTOM_PROMPT
                template_path = gemini_generate_cf_template(model, template_prompt, output_path)
            elif llm_type == "gpt":
                template_prompt = TOP_PROMPT + row["prompt"] + BOTTOM_PROMPT
                template_path = chatgpt_generate_cf_template(model, template_prompt, output_path, llm_model)
            else:   # claude
                template_prompt = TOP_PROMPT + row["prompt"] + BOTTOM_PROMPT
                template_path = claude_generate_cf_template(model, template_prompt, output_path, llm_model)
            
            # Only update the specific LLM's template path column
            df.at[index, f'{llm_type}_template_path'] = template_path
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            df.at[index, f'{llm_type}_template_path'] = 'ERROR'
    
    # Save the updated CSV
    df.to_csv(output_csv, index=False)
    return output_csv


def start_generation():
    # Configuration Variables:
    input_csv = "dataset/ioc.csv"
    llm_type = "claude"  # "gemini", "gpt" or "claude"
    llm_model = "claude-3-5-sonnet-20241022"  # [gemini-1.5-flash, gpt-4o-mini, claude-3-5-sonnet-20241022, gpt-4o, deepseek]
    output_csv = f"result/result_{llm_type}.csv"
    print(f"Start generation with {llm_type} model")

    if llm_type == "gemini":
        genai.configure(api_key=GEMIN_API_KEY)
        model = genai.GenerativeModel(llm_model)
    elif llm_type == "gpt":
        model = OpenAI(api_key=CHATGPT_API_KEY)
    elif llm_type == "claude":
        model = anthropic.Anthropic(api_key=CLAUDE_API_KEY)  # Replace with actual configuration
    elif llm_type == "deepseek":
        model = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")

    processed_csv = process_ioc_csv(input_csv, output_csv, model, llm_type, llm_model)
    print(f"Generation completed. Results saved to: {processed_csv}")


if __name__ == "__main__":
    print("Start abcd\n")
    start_generation()
    print("\nEnd")