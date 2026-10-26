import polars as pl
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

def sprint_model(t, v_max, tau): return v_max * (1 - np.exp(-t / tau))

def extract_40yd(df):
    df = df.sort("time")
    if len(df) < 25: return None
    t = np.arange(len(df)) * 0.1
    s = df["s"].to_numpy()
    if np.max(s) < 2.0: return None
    try:
        popt, _ = curve_fit(sprint_model, t, s, p0=[10.0, 0.5], bounds=([5.0, 0.1], [15.0, 2.0]))
        return {"40_v_max": popt[0], "40_tau": popt[1], "40_a_max": popt[0] / popt[1]}
    except: return None

def extract_agility(df):
    df = df.sort("time")
    if len(df) < 25: return None
    s = df["s"].to_numpy()
    dir_rad = np.radians(df["dir"].to_numpy())
    w = 7 if len(dir_rad) > 7 else len(dir_rad) - (1 - len(dir_rad) % 2)
    if w < 5: return None
    dir_smooth = savgol_filter(dir_rad, window_length=w, polyorder=2)
    ang_vel = np.abs(np.diff(dir_smooth) / 0.1)
    return {"peak_speed": np.max(s), "max_angular_vel": np.max(ang_vel) if len(ang_vel)>0 else 0}
