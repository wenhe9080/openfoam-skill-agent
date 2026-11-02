import os
from config import path_cfg, case_info
from qa_modules import QA_NoContext_deepseek_R1


SUMMARY_PROMPT = """You are an OpenFOAM engineer. After generating the corresponding OpenFOAM case files based on the user's simulation requirements, you are required to write a technical summary of the case you created:

<simulation_requirement>
{case_description}
</simulation_requirement>

<case_content>
{case_content}
</case_content>

<output_requirement>
1. Ignore settings in system/controlDict related to time stepping, meshing, post-processing, computational resources, and other efficiency-related aspects. Focus on summarizing the physical models and numerical configurations of the case.
2. Your summary should address (but is not limited to) the following points—feel free to supplement and modify as appropriate:
   - Physical background and objectives: the physical problem and goals of the simulation, and the physical models adopted.
   - Basic OpenFOAM setup: the solver employed and the flow regimes it targets (e.g., steady/unsteady, compressible/incompressible), plus the rationale behind and characteristics of the selected turbulence model (if applicable).
   - Key file descriptions: concise explanations of each file and its role.
   - Critical configuration details:
     - Initial and boundary conditions for each physical field
     - Material properties (e.g., fluid type, density, kinematic viscosity)
     - Numerical discretization schemes (e.g., differencing schemes) and solver settings
     - Special settings (e.g., multiphase models, chemical reactions, dynamic mesh, thermophysical properties, etc.)
   - Any other points that merit clarification. 
3. Formatting requirement: produce a concise technical-report-style summary that is logically rigorous and clearly articulated.{error_info}
3. Please write the entire summary in Simplified Chinese (简体中文). Solver names, turbulence model names and file names may remain in English.
</output_requirement>"""


def get_case_content(case_dir: str = path_cfg.output_dir,
                     case_name: str = case_info.case_name) -> dict:
    """
    Retrieve the contents of case files.

    Args:
        case_dir (str): Directory where the case is stored.
        case_name (str): Name of the case.

    Returns:
        dict: A dictionary containing the run result and file contents of the case.
    """
    res = {}
    for case_name_i in os.listdir(case_dir):
        case_path = os.path.join(case_dir, case_name_i)
        if not os.path.isdir(case_path) or case_name not in case_name_i:
            # print(f"Skipping non-directory item: {case_path}")
            continue

        # Determine case run status
        running_result = "success" if "cycle_index" in os.listdir(case_path) else "failed"
        res[case_name_i] = {
            "running_result": running_result,
            "case_content": {}
        }

        # Read case file contents
        case_content = {}
        for case_file in os.listdir(case_path):
            case_file_path = os.path.join(case_path, case_file)
            try:
                # Files under system/
                system_dir = os.path.join(case_file_path, "system")
                for file_name in os.listdir(system_dir):
                    file_path = os.path.join(system_dir, file_name)
                    if os.path.isfile(file_path):
                        with open(file_path) as f:
                            case_content[f"system/{file_name}"] = f.read()

                # Files under constant/
                constant_dir = os.path.join(case_file_path, "constant")
                for file_name in os.listdir(constant_dir):
                    file_path = os.path.join(constant_dir, file_name)
                    if os.path.isfile(file_path):
                        with open(file_path) as f:
                            case_content[f"constant/{file_name}"] = f.read()

                # Files under 0/ or 0.bak/
                zero_dir = os.path.join(case_file_path, "0.bak" if "0.bak" in os.listdir(case_file_path) else "0")
                for file_name in os.listdir(zero_dir):
                    file_path = os.path.join(zero_dir, file_name)
                    if os.path.isfile(file_path):
                        with open(file_path) as f:
                            content = f.read()
                            case_content[f"0/{file_name}"] = (
                                content if len(content) < 9830 else
                                content[:2000] + "\n... (content truncated due to length) ..."
                            )

                res[case_name_i]["case_content"] = case_content

            except Exception as e:
                pass
                # print(f"Error reading case files in {case_file_path}: {e}")

    return res


def generate_case_summary(case_description: str,
                          case_content: str = None,
                          error_info: str = "") -> str:
    """
    Generate a technical summary of the case.

    Args:
        case_description (str): Description of the case.
        case_content (str): Contents of the case files.
        error_info (str): Error information, if any.

    Returns:
        str: Technical summary of the case.
    """
    if case_content is None:
        case_content = "No case content provided."

    prompt = SUMMARY_PROMPT.format(
        case_description=case_description,
        case_content=case_content,
        error_info=""  # f"\n4. Error information: {error_info}" if error_info else ""
    )

    qa = QA_NoContext_deepseek_R1()
    return qa.ask(prompt)