import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

def sprint_model(t, v_max, tau):
    return v_max * (1 - np.exp(-t / tau))

def fit_sprint_kinematics(t, v):
    try:
        popt, _ = curve_fit(sprint_model, t, v, p0=[10.0, 1.0], bounds=([0, 0.01], [15, 5]))
        v_max, tau = popt
        return v_max, tau, v_max / tau
    except Exception:
        return np.nan, np.nan, np.nan

def compute_angular_velocity(direction_deg, fps=10):
    dir_rad = np.radians(direction_deg)
    dir_unwrapped = np.unwrap(dir_rad)
    if len(dir_unwrapped) < 5:
        return np.nan
    window_length = min(len(dir_unwrapped) if len(dir_unwrapped) % 2 == 1 else len(dir_unwrapped) - 1, 11)
    polyorder = min(3, window_length - 1)
    try:
        ang_vel = savgol_filter(dir_unwrapped, window_length, polyorder, deriv=1, delta=1.0/fps)
        return np.max(np.abs(ang_vel))
    except Exception:
        return np.nan
