#!/usr/bin/env python3
"""ChatCFD 端到端冒烟测试（只跑一次 case，失败则快速报错）"""
import os, sys, traceback
os.chdir(r"/mnt/g/ChatCFD-main")
sys.path.insert(0, "src")
# 先手动加载 config 让全局变量生效
import importlib, py_compile
py_compile.compile("src/config.py", doraise=True)

print("=== Step 1: import config ===")
import config
print(f"Python: {sys.version.split()[0]}")
print(f"OpenFOAM_path: {config.dependencies_cfg.OpenFOAM_path}")
print(f"OpenFOAM tutorial: {config.dependencies_cfg.OpenFOAM_tutorials_path}")
print(f"API base R1: {config.llm_cfg.DEEPSEEK_R1_BASE_URL}")
print(f"API base V3: {config.llm_cfg.DEEPSEEK_V3_BASE_URL}")

print("\n=== Step 2: load OF tutorial data ===")
import preprocess_OF_tutorial
preprocess_OF_tutorial.read_in_processed_merged_OF_cases()
print(f"Loaded {len(config.global_OF_cases)} OF cases")

print("\n=== Step 3: set a sample case and run ===")
from datetime import datetime
config.path_cfg.document_path = "/mnt/g/ChatCFD-main/pdf/sun_2023_naca0012.pdf"
config.path_cfg.grid_path     = "/mnt/g/ChatCFD-main/grids/naca0012.msh"
config.run_cfg.run_time       = 1
config.case_info.file_name    = "sample_naca0012"
config.path_cfg.output_path   = os.path.join(config.path_cfg.output_dir, "sample_naca0012")
config.ensure_directory_exists(config.path_cfg.output_path)

# Set minimal case info
config.case_info.case_name     = "sample_NACA0012_AOA10_kOmegaSST"
config.case_info.case_solver   = "simpleFoam"
config.case_info.turbulence_model = "kOmegaSST"
config.case_info.other_physical_model = None
config.case_info.case_description = "2D incompressible flow around NACA0012 airfoil at AOA=10 degrees"
config.case_info.simulation_requirement = config.case_info.case_description
config.case_info.case_info_simple = {
    "Case_1": {
        "case_name": "sample_NACA0012_AOA10_kOmegaSST",
        "solver": "simpleFoam",
        "turbulence_model": "kOmegaSST",
        "other_physical_model": None,
        "case_specific_description": "2D incompressible flow around NACA0012 airfoil at AOA=10 degrees"
    }
}

print("\n=== Step 4: run main_run_chatcfd.run_case() ===")
import main_run_chatcfd
try:
    results = main_run_chatcfd.run_case()
    print(f"RUN RESULTS: {results}")
except Exception as e:
    print(f"RUN FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n=== DONE ===")
