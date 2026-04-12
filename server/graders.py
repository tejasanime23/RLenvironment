def clamp_score(score):
    """
    Ensures scores stay strictly within (0, 1) to satisfy platform rules.
    1.0 becomes 0.9999, 0.0 becomes 0.0001.
    """
    try:
        f_score = float(score)
        return max(0.0001, min(0.9999, f_score))
    except (ValueError, TypeError):
        return 0.0001 # Default fallback for invalid data

def _extract_meta(history_or_info):
    if isinstance(history_or_info, dict):
        return history_or_info.get("metadata", {"critical_path_depth": 18, "total_ops": 177, "max_area": 1000.0})
    return {"critical_path_depth": 18, "total_ops": 177, "max_area": 1000.0}

def grade_task_1(history_or_info):
    """ Task 1: Infinite ALUs. Linear drop-off from Critical Path """
    cycles = history_or_info.get("global_state", [999])[0] if isinstance(history_or_info, dict) else 999
    target = _extract_meta(history_or_info)["critical_path_depth"]
    
    score = 1.0 - (max(0, cycles - target) * 0.05)
    return clamp_score(score)

def grade_task_2(history_or_info):
    """ Task 2: Strict Hardware Bounds. Asymptotic Decay targeting Bottleneck threshold """
    cycles = history_or_info.get("global_state", [999])[0] if isinstance(history_or_info, dict) else 999
    meta = _extract_meta(history_or_info)
    
    # Add +10 cycles of 'slack' to account for small-kernel overhead and pipeline drain
    target = max(meta["critical_path_depth"], float(meta["total_ops"]) / 2.0) + 10.0
    
    if cycles <= target:
        return clamp_score(1.0)
    # Balanced linear decay for realistic agent performance on constrained tasks
    score = target / float(cycles)
    return clamp_score(score)

def grade_task_3(history_or_info):
    """ Task 3: Pragma Scaling. Accelerated Speed multiplied by Area Efficiency """
    global_s = history_or_info.get("global_state", [999, 0, 0, 1.0]) if isinstance(history_or_info, dict) else [999, 0, 0, 1.0]
    cycles = global_s[0]
    area_ratio = global_s[3]
    
    meta = _extract_meta(history_or_info)
    target = 20.0 # Hard accelerated gold standard (user defined ceiling)
    
    if cycles <= target:
        speed_score = 1.0
    else:
        speed_score = (target / float(cycles)) ** 2
        
    area_efficiency = 1.2 - (0.2 * float(area_ratio))
    score = speed_score * area_efficiency
    return clamp_score(score)
