from checkov.cloudformation.runner import Runner
from checkov.runner_filter import RunnerFilter
from checkov.common.output.report import Report
import shutil
import os
import pandas as pd
import time
from evaluation.cloud_evaluation import analyze_resource_coverage


def process_checkov_validation(input_csv: str, output_csv: str, start_row: int, end_row: int):
    """
    Not Used
    Process records from CSV file to validate templates using Checkov.
    
    Args:
        input_csv: Path to input CSV file containing template information
        output_csv: Path to output CSV file for validation results
        start_row: Starting row index for processing
        end_row: Ending row index for processing (exclusive)
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_csv)
        
        # Initialize new columns for validation results
        df['checkov_pass_user_intent'] = None
        df['checkov_validation_details'] = None
        
        # Process rows within the specified range
        for idx in range(start_row, min(end_row, len(df))):
            try:
                # Get template path and user intent file
                template_path = df.loc[idx, 'final_template_path']
                user_intent_file = df.loc[idx, 'user_intent']
                user_intent_id = df.loc[idx, 'user_intent_id']
                
                # Validate template using Checkov
                result = validate_with_checkov_package(
                    template_path=template_path,
                    user_intent_file=user_intent_file,
                    user_intent_id=user_intent_id
                )
                
                # Store results in DataFrame
                df.loc[idx, 'checkov_pass_user_intent'] = result['pass_user_intent']
                df.loc[idx, 'checkov_validation_details'] = str(result['details'])
                
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                df.loc[idx, 'checkov_pass_user_intent'] = False
                df.loc[idx, 'checkov_validation_details'] = f"Error: {str(e)}"
        
        # Save results to output CSV
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
        
    except Exception as e:
        print(f"Error processing CSV file: {str(e)}")


def make_temp_dir(user_intent_file: str) -> str:
    """
    Make a temporary directory for the user intent validation file in the same location as the input file.
    
    Args:
        user_intent_file: Path to the user intent validation file
        
    Returns:
        str: Path to the temporary directory containing the copied validation file
    """
    try:
        # Get the directory where user_intent_file is located
        parent_dir = os.path.dirname(os.path.abspath(user_intent_file))
        
        # Create a temporary directory name
        temp_dir_name = f"temp_checkov_{os.path.splitext(os.path.basename(user_intent_file))[0]}"
        temp_dir = os.path.join(parent_dir, temp_dir_name)
        
        # Create the directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)
        
        # Get the filename from the path
        filename = os.path.basename(user_intent_file)
        
        # Copy the file to the temporary directory
        dest_path = os.path.join(temp_dir, filename)
        shutil.copy2(user_intent_file, dest_path)
        
        return temp_dir
        
    except Exception as e:
        # Clean up the temporary directory if something goes wrong
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise Exception(f"Failed to create temporary directory: {str(e)}")
    

def validate_with_checkov_package(template_path: str, user_intent_files: list[str] = None, user_intent_ids: list[str] = None) -> dict:
    """
    Validate a CloudFormation template using Checkov's Python package directly, focusing only on user intent validation.
    
    Args:
        template_path: Path to the CloudFormation template
        user_intent_files: List of paths to the custom policy files
        user_intent_ids: List of IDs of the custom user intent checks
        
    Returns:
        dict: Results of user intent validation
    """
    temp_dir = None
    runner = None
    try:
        # Create temporary directory for custom policy if provided
        external_checks_dir = None
        if user_intent_files:
            if os.path.isfile(user_intent_files[0]):
                temp_dir = make_temp_dir(user_intent_files[0])
                external_checks_dir = temp_dir
                # Copy all user intent files to the temporary directory
                for user_intent_file in user_intent_files[1:]:
                    if os.path.isfile(user_intent_file):
                        filename = os.path.basename(user_intent_file)
                        dest_path = os.path.join(temp_dir, filename)
                        shutil.copy2(user_intent_file, dest_path)
            else:
                external_checks_dir = user_intent_files

        # Create a new runner instance for each validation
        runner = Runner()
        runner_filter = RunnerFilter(
            framework=['cloudformation'],
            skip_checks=['CKV*']  # Skip all default security checks
        )

        # Run the checks
        report = runner.run(
            root_folder=None,
            external_checks_dir=[external_checks_dir] if external_checks_dir else None,
            files=[template_path],
            runner_filter=runner_filter,
        )

        # Process results
        failed_checks = []
        passed_checks = []

        for record in report.failed_checks:
            failed_checks.append({
                'check_id': record.check_id,
                'check_name': record.check_name,
                'file_path': record.file_path,
                'resource': record.resource,
                'guideline': record.guideline,
            })

        for record in report.passed_checks:
            passed_checks.append({
                'check_id': record.check_id,
                'check_name': record.check_name,
                'file_path': record.file_path,
                'resource': record.resource,
            })

        if user_intent_ids:
            unique_passed_check_ids = set(check['check_id'] for check in passed_checks)
            pass_user_intent = True
            for user_intent_id in user_intent_ids:
                if user_intent_id not in unique_passed_check_ids:
                    pass_user_intent = False
                    break
        else:
            pass_user_intent = False

        return {
            "pass_user_intent": pass_user_intent,
            "details": {
                "passed_checks": passed_checks,
                "failed_checks": failed_checks
            }
        }

    except Exception as e:
        print(f"Check this row + {str(e)}")
        return {
            "success": False,
            "error": f"Error running Checkov: {str(e)}",
        }
    
    finally:
        # Clean up the temporary directory if it was created
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)



def process_templates(result_csv_path, iac_user_intent_csv_path, output_csv_path):
    """
    Process templates and validate them using Checkov and resource coverage analysis.
    
    Args:
        result_csv_path: Path to the result CSV file containing template paths
        iac_user_intent_csv_path: Path to the IAC user intent CSV file
        output_csv_path: Path to save the output results
    """
    try:
        # Read the CSV files
        result_df = pd.read_csv(result_csv_path)
        iac_df = pd.read_csv(iac_user_intent_csv_path)
        
        # Filter rows where user_intent is not empty
        iac_df = iac_df[iac_df['user_intent'].notna()]
        
        # Initialize results list
        results = []
        
        # Process each row
        for _, row in iac_df.iterrows():   # row 2
            print(row['row_number'])
            row_number = row['row_number']

            try:   # Handle the case where the result file is empty or partially completed
                template_path = result_df.loc[row_number, 'final_template_path']
            except Exception as e:
                print(f"Row {row_number} has no template path")
                break

            ground_truth_path = row['ground_truth_path']
            user_intent_files = row['user_intent_file_path'].split(', ') if pd.notna(row['user_intent_file_path']) else None
            user_intent_ids = row['user_intent_id'].split(', ') if pd.notna(row['user_intent_id']) else None
            
            # Validate with Checkov
            checkov_result = validate_with_checkov_package(
                template_path=template_path,
                user_intent_files=user_intent_files,
                user_intent_ids=user_intent_ids
            )
            
            # Analyze resource coverage
            coverage_result = analyze_resource_coverage(ground_truth_path, template_path)
            
            # Compile results
            result = {
                'row_number': row_number,
                'template_path': template_path,
                'ground_truth_path': ground_truth_path,
                'pass_user_intent': checkov_result['pass_user_intent'],
                'checkov_details': checkov_result['details'],
                'coverage_percentage': coverage_result['coverage_percentage'],
                'accuracy_percentage': coverage_result['accuracy_percentage'],
                'resource_details': coverage_result['resource_details'],
                'missing_resources': coverage_result['resource_details']['missing'],
            }
            results.append(result)
        
        # Create DataFrame from results
        results_df = pd.DataFrame(results)
        
        # Save results to CSV
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        results_df.to_csv(output_csv_path, index=False)
        print(f"Results saved to {output_csv_path}")
        
    except Exception as e:
        print(f"Error processing files: {str(e)}")


if __name__ == "__main__":
    IAC_USER_INTENT_CSV_PATH = "Data/iac_user_intent.csv"

    llm_model = "claude-3-7-sonnet-20250219"   # You only need to change this before run the file. Note you should ensure you ran main.py with this llm model.
    result_csv_path = f"Result/iterative_{llm_model}_results.csv"
    output_csv_path = f"Result/user_intent/user_intent_{llm_model}_results.csv"

    print("Start Checkov Validation")
    process_templates(result_csv_path, IAC_USER_INTENT_CSV_PATH, output_csv_path)
    print("End Checkov Validation")
