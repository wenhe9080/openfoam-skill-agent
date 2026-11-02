import os
import json
import requests
import config
import re

# sub-folders in the OpenFOAM tutorial
solver_features = ['basic', 'compressible', 'heatTransfer', 'lagrangian',
                  'multiphase', 'DNS', 'combustion', 'incompressible', 'combustion']

# Specify subdirectories to collect
target_subdirs = {'0', '0.orig', 'system', 'constant'}

# Dictionary to store results
cases_dict_collection = {}

# LLM call

# Remove cases with overly long configuration files
def describe_cases(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
        
    # Define a recursive function to traverse nested dictionary structure
    def recursive_process(data):
        if isinstance(data, dict):
            for key, value in data.items():
                # Check if 'case_path' and 'configuration_files' exist
                if 'case_path' in value and 'configuration_files' in value:
                    case_path = value['case_path']
                    # Convert 'configuration_files' dictionary to string
                    config_str = json.dumps(value['configuration_files'], ensure_ascii=False)

                    # print("Processing case = ",case_path)
                    
                    # Skip cases that are too long
                    if(len(config_str) > 2e5):
                        # print(f"Removing {case_path} since its too long. len(config_str) = {len(config_str)}")
                        continue
                    
                    # Append processed data to output file
                    with open(output_file, 'a', encoding='utf-8') as outfile:
                        json.dump({key: value}, outfile, ensure_ascii=False, indent=4)
                        outfile.write('\n')
                else:
                    # Recursively process nested parts
                    recursive_process(value)
        elif isinstance(data, list):
            for item in data:
                recursive_process(item)

    recursive_process(data)


# Collect case description files from openfoam/tutorial directory
def case_config_collector():
    for feature in solver_features:
        feature_dir = os.path.join(config.of_tutorial_dir, feature)
        if not os.path.isdir(feature_dir):
            continue  # Skip if feature directory doesn't exist
        for solver in os.listdir(feature_dir):
            solver_dir = os.path.join(feature_dir, solver)
            if os.path.isdir(solver_dir) and solver.endswith('Foam'):
                for root_path, dirs, files in os.walk(solver_dir):
                    if 'system' in dirs:
                        system_dir = os.path.join(root_path, 'system')
                        required_files = {'controlDict', 'fvSchemes', 'fvSolution'}
                        if os.path.isdir(system_dir):
                            system_files = set(os.listdir(system_dir))
                            if required_files.issubset(system_files):
                                # Get relative path of the case
                                case_relative_path = os.path.relpath(root_path, config.of_tutorial_dir)
                                # Initialize storage structure
                                cases_dict_collection.setdefault(feature, {}).setdefault(solver, {})[case_relative_path] = {
                                    'case_path': case_relative_path,
                                    'configuration_files': {}
                                }
                                # Collect configuration files and their contents from specified subdirectories
                                config_files = {}
                                for subdir in target_subdirs:
                                    subdir_path = os.path.join(root_path, subdir)
                                    if os.path.isdir(subdir_path):
                                        for dirpath, dirnames, filenames in os.walk(subdir_path):
                                            # If current directory is constant, skip polyMesh subdirectory
                                            if os.path.basename(subdir_path) == 'constant':
                                                if 'polyMesh' in dirnames:
                                                    dirnames.remove('polyMesh')  # Remove 'constant/polyMesh' from dirnames to prevent os.walk from traversing it
                                            if os.path.basename(subdir_path) == '0':
                                                if 'include' in dirnames:
                                                    dirnames.remove('include')  # Remove '0/include' from dirnames to prevent os.walk from traversing it
                                            for filename in filenames:
                                                if "blockMeshDict" in filename:
                                                    continue
                                                if "changeDictionaryDict" in filename:
                                                    continue
                                                file_full_path = os.path.join(dirpath, filename)
                                                # Get file path relative to case directory
                                                file_relative_path = os.path.relpath(file_full_path, root_path)
                                                # Read file content
                                                try:
                                                    with open(file_full_path, 'r', encoding='utf-8', errors='ignore') as f:
                                                        # Read file and remove header information
                                                        lines = f.readlines()
                                                        content_started = False
                                                        processed_lines = []
                                                        for line in lines:
                                                            if not content_started and 'FoamFile' in line:
                                                                content_started = True
                                                            if content_started:
                                                                processed_lines.append(line)
                                                        # Merge processed content into string
                                                        file_content = ''.join(processed_lines)
                                                except Exception as e:
                                                    print(f"Cannot read file {file_full_path}, error: {e}")
                                                    file_content = ""
                                                # Add file path and content to configuration files dictionary
                                                config_files[file_relative_path] = file_content
                                # Update configuration files list
                                cases_dict_collection[feature][solver][case_relative_path]['configuration_files'] = config_files

def merge_json_objects(file_path, output_path):
    # Read file content
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split multiple JSON objects
    json_strings = content.split('}\n{')
    
    # Create merged dictionary
    merged_dict = {}
    
    for i, json_str in enumerate(json_strings):
        # Add missing brackets for split parts
        if i > 0:
            json_str = '{' + json_str
        if i < len(json_strings) - 1:
            json_str = json_str + '}'
            
        try:
            # Parse JSON
            data = json.loads(json_str)
            # Merge into main dictionary
            merged_dict.update(data)
        except json.JSONDecodeError as e:
            print(f"Error parsing {i+1}th JSON object: {str(e)}")
    
    # Write merged data to new file
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(merged_dict, file, indent=4)
    
    return merged_dict

multiphase_flow_solvers = [
    "cavitatingFoam","compressibleInterFoam","compressibleMultiphaseInterFoam",
    "driftFluxFoam","icoReactingMultiphaseInterFoam","interCondensatingEvaporatingFoam",
    "interFoam","interIsoFoam","interPhaseChangeFoam","multiphaseEulerFoam",
    "multiphaseInterFoam","potentialFreeSurfaceFoam","twoLiquidMixingFoam",
    "chtMultiRegionFoam","reactingMultiphaseEulerFoam","reactingTwoPhaseEulerFoam","twoPhaseEulerFoam"
]

particle_flow_solvers = [
    "denseParticleFoam","particleFoam","reactingParcelFoam",
    "DPMFoam","MPPICFoam","icoUncoupledKinematicParcelDyMFoam",
    "kinematicParcelFoam","sprayFoam","MPPICDyMFoam",
    "coalChemistryFoam","icoUncoupledKinematicParcelFoam",
    "reactingHeterogenousParcelFoam","simpleReactingParcelFoam",
    "uncoupledKinematicParcelDyMFoam"
]

reacting_flow_solvers = [
    "buoyantReactingFoam","chemFoam","coldEngineFoam",
    "reactingFoam","fireFoam","PDRFoam","XiFoam",
    "XiEngineFoam", "rhoReactingFoam"
]

solver_set = set()
turbulence_type_set = set()
turbulence_model_set = set()
boundary_type_set = set()

def extract_turbulence_model(file_content):
    content = file_content.split('\n')
    model = None
    for line in content:
        if 'model' in line.lower():
            model = line.split()[-1].strip(';')
            break  # Exit loop once we've found the model
    return model

def add_case_path_keys(data):
    for case_data in data.values():
        config_files = case_data["configuration_files"]

        # print(f"updating case: {case_data['case_path']}")

        # Step 1: Process 0.orig/* keys and rename to 0/*
        keys_to_modify = []
        for key in list(config_files.keys()):
            if key.startswith("0.orig/"):
                new_key = key.replace("0.orig/", "0/", 1)
                # Remove .orig suffix in filename if present
                parts = new_key.split('/')
                if len(parts) > 1 and parts[-1].endswith('.orig'):
                    parts[-1] = parts[-1][:-5]  # Remove .orig extension
                    new_key = '/'.join(parts)
                keys_to_modify.append((key, new_key))
        
        # Perform key replacement
        for old_key, new_key in keys_to_modify:
            if old_key in config_files:
                config_files[new_key] = config_files.pop(old_key)
        
        # Step 2: Collect all 0/* keys for required_field
        required_fields = [k for k in config_files if k.startswith("0/")]
        case_data["required_field"] = required_fields
        
        # Extract solver (application)
        control_dict = config_files.get("system/controlDict", "")
        solver_match = re.search(r"application\s+(\w+);", control_dict)
        solver = solver_match.group(1) if solver_match else None
        case_data["solver"] = solver

        solver_set.add(solver)
        
        # Determine singlePhase
        case_data["singlePhase"] = True
        if solver in multiphase_flow_solvers:
            case_data["singlePhase"] = False
        
        # Determine particle flow
        case_data["particle_flow"] = False
        if case_data["particle_flow"] == False:
            if solver in particle_flow_solvers:
                case_data["particle_flow"] = True
            if any('Cloud' in s for s in list(config_files.keys())):
                case_data["particle_flow"] = True

        # Determine reacting flow
        case_data["reacting_flow"] = False
        if case_data["reacting_flow"] == False:
            if solver in reacting_flow_solvers:
                case_data["reacting_flow"] = True
            if "constant/combustionProperties" in list(config_files.keys()):
                case_data["reacting_flow"] = True
            if "constant/reactions" in list(config_files.keys()):
                case_data["reacting_flow"] = True

        # Extract turbulence_type (RAS, LES, laminar ...)
        # Extract turbulence_model (kEpsilon ...)

        turbulence_type = None
        turbulence_model = None
        for file_path in config_files:
            parts = file_path.split("/")
            if len(parts) > 1 and parts[0] == "constant" and parts[-1] == "turbulenceProperties":
                content = config_files[file_path]
                type_match = re.search(r"simulationType\s+(\w+);", content)
                if type_match:
                    turbulence_type = type_match.group(1)
                    if turbulence_type == "LES":
                        type_match = re.search(r"LESModel\s+(\w+);", content)
                        if type_match:
                            turbulence_model = type_match.group(1)
                    elif turbulence_type == "RAS":
                        type_match = re.search(r"RASModel\s+(\w+);", content)
                        if type_match:
                            turbulence_model = type_match.group(1)
                    elif turbulence_type == "laminar":
                        turbulence_model = None
                    break

        case_data["turbulence_type"] = turbulence_type
        case_data["turbulence_model"] = turbulence_model
        

        turbulence_type_set.add(turbulence_type)
        turbulence_model_set.add(turbulence_model)

        # Extract boundary types from case
        case_boundary_type_set = set()
        for file_path in config_files:
            parts = file_path.split("/")
            if len(parts) > 1 and (parts[0] == "0" or parts[0] == "0.org"):
                content = config_files[file_path]

                # Step 1: Match content of {} block after boundaryField
                boundary_field_pattern = re.compile(
                    r'boundaryField\s*{((?:[^{}]*{[^{}]*}[^{}]*)*)}', 
                    re.DOTALL
                )
                boundary_match = boundary_field_pattern.search(content)
                if boundary_match:
                    boundary_content = boundary_match.group(1)
                    
                    # Step 2: Match all values after type
                    type_pattern = re.compile(r'type\s+([^;]+);', re.DOTALL)
                    type_matches = type_pattern.findall(boundary_content)

                    if type_matches:
                        # Remove leading and trailing whitespace and output results
                        type_values = [m.strip() for m in type_matches]
                        case_boundary_type_set.update(type_values)

        case_data["boundary_type"] = list(case_boundary_type_set)
        boundary_type_set.update(case_boundary_type_set)

    return data


def main():

    # Collect all case files from tutorial into a json file
    case_config_collector()
    all_case_collector = f'{config.Database_OFv24_PATH}/openfoam_cases.json'
    with open(all_case_collector, 'w', encoding='utf-8') as f:
        json.dump(cases_dict_collection, f, indent=4, ensure_ascii=False)

    discrete_tmp_json = f'{config.Database_OFv24_PATH}/discrete_case_config_with_descriptions.json'

    describe_cases(all_case_collector, output_file=discrete_tmp_json)

    output_file = f'{config.Database_OFv24_PATH}/merged_OF_cases.json'

    merge_json_objects(discrete_tmp_json, output_file)

    # os.remove(all_case_collector)
    # os.remove(discrete_tmp_json)

    # Read input JSON
    with open(output_file, "r") as f:
        data = json.load(f)
        print("Total case number = ", len(data))

    # Process data
    updated_data = add_case_path_keys(data)

    new_solver_set = {item for item in solver_set if item is not None}
    new_turbulence_type_set = {item for item in turbulence_type_set if item is not None}
    new_turbulence_model_set = {item for item in turbulence_model_set if item is not None}
    new_boundary_type_set = {item for item in boundary_type_set if item is not None}

    config.global_OF_keywords = {
        "solver": list(new_solver_set),
        "turbulence_type": list(new_turbulence_type_set),
        "turbulence_model": list(new_turbulence_model_set),
        "boundary_type": list(new_boundary_type_set)
    }

    ofv24_keywords_file = f'{config.Database_OFv24_PATH}/ofv24_keywords.json'

    with open(ofv24_keywords_file, 'w', encoding='utf-8') as file:
        json.dump(config.global_OF_keywords, file, indent=4)

    processed_merged_OF_cases = f'{config.Database_OFv24_PATH}/processed_merged_OF_cases.json'

    # Output results
    with open(processed_merged_OF_cases, "w") as f:
        json.dump(updated_data, f, indent=4)

    config.global_OF_cases = updated_data
    config.flag_tutorial_preprocessed = True

def read_in_processed_merged_OF_cases():
    # If not running preprocess, read case data from previously run processed_merged_OF_cases.json into config.global_OF_cases
    proprocess_case_json_file = f"{config.Database_OFv24_PATH}/processed_merged_OF_cases.json"
    with open(proprocess_case_json_file, 'r', encoding='utf-8') as file:
        # Read JSON file content into a dictionary
        config.global_OF_cases = json.load(file)
    # Update config.global_OF_keywords
    solver_set = set()
    turbulence_type_set = set()
    turbulence_model_set = set()
    boundary_type_set = set()

    for key,value in config.global_OF_cases.items():
        solver_set.add(value["solver"])
        turbulence_type_set.add(value["turbulence_type"])
        turbulence_model_set.add(value["turbulence_model"])
        boundary_type_set.update(value["boundary_type"])

    new_solver_set = {item for item in solver_set if item is not None}
    new_turbulence_type_set = {item for item in turbulence_type_set if item is not None}
    new_turbulence_model_set = {item for item in turbulence_model_set if item is not None}
    new_boundary_type_set = {item for item in boundary_type_set if item is not None}

    config.global_OF_keywords = {
        "solver": list(new_solver_set),
        "turbulence_type": list(new_turbulence_type_set),
        "turbulence_model": list(new_turbulence_model_set),
        "boundary_type": list(new_boundary_type_set)
    }

    a = 1