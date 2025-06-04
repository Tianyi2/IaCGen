from checkov.cloudformation.runner import Runner
from checkov.runner_filter import RunnerFilter
import shutil
import os
import pandas as pd


def validate_security_with_checkov_package(template_path: str) -> dict:
    """
    Validate a CloudFormation template using Checkov's Python package directly.
    
    Args:
        template_path: Path to the CloudFormation template
        
    Returns:
        dict: Results of validation including both user intent and security checks
    """
    temp_dir = None
    try:  
        # Initialize the runner
        runner = Runner()
        runner_filter = RunnerFilter(
            framework=['cloudformation'],
        )

        # Run the checks
        report = runner.run(
            root_folder=None,
            # external_checks_dir=[external_checks_dir] if external_checks_dir else None,
            files=[template_path],
            runner_filter=runner_filter,
        )

        # Process results
        failed_checks = []
        passed_checks = []

        for record in report.failed_checks:
            failed_checks.append({
                'check_id': record.check_id,    # id of the check file under meta
                'check_name': record.check_name,   # name of the check file under meta
                'file_path': record.file_path,   # The being checked file path
                'resource': record.resource,   # The resource in the temple (i.e., AWS::EC2::SecurityGroup.InstanceSecurityGroup)
                'guideline': record.guideline,
            })

        for record in report.passed_checks:
            passed_checks.append({
                'check_id': record.check_id,
                'check_name': record.check_name,
                'file_path': record.file_path,
                'resource': record.resource,
            })

        return process_checkov_result(passed_checks, failed_checks)

    except Exception as e:
        return {
            "success": False,
            "error": f"Error running Checkov: {str(e)}",
        }
    
    finally:
        # Clean up the temporary directory if it was created
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def process_checkov_result(passed_checks, failed_checks):
    """
    Process the checkov result:
    First, check if the template pass user intent check (if pass the external check).
    Second, check if the template pass CheckOv default security checks. 
    
    Args:
        passed_checks: List of passed check records from Checkov
        failed_checks: List of failed check records from Checkov
        
    Returns:
        dict: Results containing:
            - pass_security_check: Boolean indicating if all security checks passed
            - security_check_details: Details about security check results
    """   
    # Process security checks (excluding user intent check)
    security_failed_checks = [
        check for check in failed_checks 
    ]
    
    security_passed_checks = [
        check for check in passed_checks 
    ]
    
    # Calculate security check metrics
    total_security_checks = len(security_passed_checks) + len(security_failed_checks)
    pass_percentage = (
        round(len(security_passed_checks) / total_security_checks * 100, 2)
        if total_security_checks > 0 
        else 0
    )
    
    # Get details of failed security checks
    failed_check_details = [
        {
            'id': check['check_id'],
            'name': check['check_name'],
            'resource': check['resource']
        }
        for check in security_failed_checks
    ]
    
    return {
        "pass_security_check": len(security_failed_checks) == 0,
        "security_check_details": {
            "total_checks": total_security_checks,
            "passed_checks": len(security_passed_checks),
            "failed_checks": len(security_failed_checks),
            "pass_percentage": pass_percentage,
            "failed_check_details": failed_check_details
        }
    }


def process_security_validation_with_checkov(input_csv: str, output_csv: str, start_row: int, end_row: int):
    """
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
        
        # If output_csv exists, read it and update only the specified columns for the processed rows
        if os.path.exists(output_csv):
            out_df = pd.read_csv(output_csv)
            # Ensure the output DataFrame has the necessary columns
            for col in ['checkov_pass_security', 'checkov_pass_security_percentage', 'checkov_security_details']:
                if col not in out_df.columns:
                    out_df[col] = None
        else:
            out_df = df.copy()
            out_df['checkov_pass_security'] = None
            out_df['checkov_pass_security_percentage'] = None
            out_df['checkov_security_details'] = None
        
        # Process rows within the specified range
        for idx in range(start_row, min(end_row, len(df))):
            try:
                # Get template path and user intent file
                template_path = df.loc[idx, 'final_template_path']   # the column where contain the template file path
                
                # Validate template using Checkov
                result = validate_security_with_checkov_package(
                    template_path=template_path,
                )
                print(result)
                # Store results in DataFrame (update only the processed rows)
                out_df.loc[idx, 'checkov_pass_security'] = result['pass_security_check']
                out_df.loc[idx, 'checkov_pass_security_percentage'] = result['security_check_details']['pass_percentage']
                out_df.loc[idx, 'checkov_security_details'] = str(result['security_check_details'])
                
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                out_df.loc[idx, 'checkov_pass_security'] = False
                out_df.loc[idx, 'checkov_pass_security_percentage'] = 0
                out_df.loc[idx, 'checkov_security_details'] = f"Error: {str(e)}"
        
        # Save results to output CSV (appending/updating only the relevant columns)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        out_df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
        
    except Exception as e:
        print(f"Error processing CSV file: {str(e)}")


# Start
if __name__ == "__main__":
    input_csv = "Result/iterative_claude-3-7-sonnet-20250219_results.csv"
    llm_model = input_csv.split("_")[1]
    output_csv = f"Result/security/security_{llm_model}_results.csv"
    start_row = 0
    end_row = 153

    print("Start Checkov Validation")
    process_security_validation_with_checkov(input_csv, output_csv, start_row, end_row)
    print("End Checkov Validation")
