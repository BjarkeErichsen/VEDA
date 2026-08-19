# sampling_config.py

dpm_solver_config = {
    "steps": 10,  # DPM-Solver 通常在 10-20 步内收敛
    "order": 2,  # 二阶 DPM-Solver
    "skip_type": "time_quadratic",#logSNR, time_uniform, time_quadratic
    "method": "multistep", # adaptive, singlestep_fixed, multistep, singlestep
    "denoise_to_zero": True,
    "atol": 1e-1,
    "rtol": 5e-1
}

ddim_config = {
    "steps": 500,  # DDIM 采样步数通常较少
    "eta": 0.5,  # η=0 对应确定性 DDIM 采样
    "method": "ddim"
}
edm_dpm_solver_config = {
    # "steps": 15,  # DPM-Solver 通常在 10-20 步内收敛
    "order": 2,  # 二阶 DPM-Solver
    "skip_type": "logSNR",#logSNR, time_uniform, time_quadratic
    "method": "multistep", # adaptive, singlestep_fixed, multistep, singlestep
    "denoise_to_zero": False,
    # "atol": 1e-1,
    # "rtol": 5e-1
}