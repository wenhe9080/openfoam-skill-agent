#!/usr/bin/env python3
"""ChatCFD 复现驱动脚本 2：可压缩 SERN 喷管案例（Yu_2023_nozzle.pdf）
论文参数：2D 过膨胀单边膨胀喷管(SERN)，NPR≈3，理想气体，
入口总压 93243.66 Pa @300K，环境背压 31073.02 Pa @300K，
求解器 rhoCentralFoam，湍流模型 RNG k-epsilon（论文 Fluent 参考设置）。
"""
import os, sys, traceback
os.environ["HF_HUB_OFFLINE"] = "1"          # use local HF cache, skip network HEAD checks
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.chdir(r"/mnt/g/ChatCFD-main")
sys.path.insert(0, "src")

print("=== Step 1: import config ===")
import config
print(f"Python: {sys.version.split()[0]}")
print(f"OpenFOAM_path: {config.dependencies_cfg.OpenFOAM_path}")
print(f"API base R1: {config.llm_cfg.DEEPSEEK_R1_BASE_URL}")
print(f"Model R1: {os.environ.get('DEEPSEEK_R1_MODEL_NAME')}")

print("\n=== Step 2: load OF tutorial data ===")
import preprocess_OF_tutorial
preprocess_OF_tutorial.read_in_processed_merged_OF_cases()
print(f"Loaded {len(config.global_OF_cases)} OF cases")

print("\n=== Step 3: set nozzle case and run ===")
config.path_cfg.document_path = "/mnt/g/ChatCFD-main/pdf/Yu_2023_nozzle.pdf"
config.path_cfg.grid_path     = "/mnt/g/ChatCFD-main/grids/Yu_2023_nozzle.msh"
config.run_cfg.run_time       = 1
config.case_info.file_name    = "sample_nozzle"
config.path_cfg.output_path   = os.path.join(config.path_cfg.output_dir, "sample_nozzle")
config.ensure_directory_exists(config.path_cfg.output_path)

# Case info from the paper (Fluent reference setup in Table: NPR=3)
config.case_info.case_name     = "sample_SERN_NPR3_rhoCentralFoam"
config.case_info.case_solver   = "rhoCentralFoam"
config.case_info.turbulence_model = "RNGkEpsilon"
config.case_info.other_physical_model = None
config.case_info.case_description = ("2D compressible overexpanded single expansion ramp nozzle (SERN) flow "
    "at nozzle pressure ratio NPR=3. Ideal gas. Nozzle inlet total pressure 93243.66 Pa, total temperature 300 K. "
    "Ambient/far-field static pressure 31073.02 Pa, temperature 300 K.")
config.case_info.simulation_requirement = config.case_info.case_description
config.case_info.case_info_simple = {
    "Case_1": {
        "case_name": "sample_SERN_NPR3_rhoCentralFoam",
        "solver": "rhoCentralFoam",
        "turbulence_model": "RNGkEpsilon",
        "other_physical_model": None,
        "case_specific_description": config.case_info.case_description
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
