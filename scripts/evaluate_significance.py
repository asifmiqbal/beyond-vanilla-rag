import json
import numpy as np
from scipy import stats

with open('expanded_multidomain_results.json') as f:
    raw = json.load(f)

# Collect matched pairs across all 36 scenarios
lat_17b_ctrl = [v['streams']['S1_1.7B_Ctrl']['median_ms'] for v in raw.values()]
lat_17b_regex = [v['streams']['S2_1.7B_Regex']['median_ms'] for v in raw.values()]
lat_17b_jev = [v['streams']['S3_1.7B_Jev']['median_ms'] for v in raw.values()]

rec_17b_ctrl = [v['streams']['S1_1.7B_Ctrl']['grounding_pct'] for v in raw.values()]
rec_17b_regex = [v['streams']['S2_1.7B_Regex']['grounding_pct'] for v in raw.values()]
rec_17b_jev = [v['streams']['S3_1.7B_Jev']['grounding_pct'] for v in raw.values()]

lat_7b_ctrl = [v['streams']['S4_7B_Ctrl']['median_ms'] for v in raw.values()]
lat_7b_regex = [v['streams']['S5_7B_Regex']['median_ms'] for v in raw.values()]
lat_7b_jev = [v['streams']['S6_7B_Jev']['median_ms'] for v in raw.values()]

rec_7b_ctrl = [v['streams']['S4_7B_Ctrl']['grounding_pct'] for v in raw.values()]
rec_7b_regex = [v['streams']['S5_7B_Regex']['grounding_pct'] for v in raw.values()]
rec_7b_jev = [v['streams']['S6_7B_Jev']['grounding_pct'] for v in raw.values()]

eff_7b_ctrl = [v['streams']['S4_7B_Ctrl']['token_efficiency'] for v in raw.values()]
eff_7b_jev = [v['streams']['S6_7B_Jev']['token_efficiency'] for v in raw.values()]

eff_17b_ctrl = [v['streams']['S1_1.7B_Ctrl']['token_efficiency'] for v in raw.values()]
eff_17b_jev = [v['streams']['S3_1.7B_Jev']['token_efficiency'] for v in raw.values()]

t_lat_17, p_lat_17 = stats.ttest_rel(lat_17b_ctrl, lat_17b_jev)
t_lat_7, p_lat_7 = stats.ttest_rel(lat_7b_ctrl, lat_7b_jev)
t_rec_17, p_rec_17 = stats.ttest_rel(rec_17b_jev, rec_17b_regex)
t_eff_7, p_eff_7 = stats.ttest_rel(eff_7b_jev, eff_7b_ctrl)
t_eff_17, p_eff_17 = stats.ttest_rel(eff_17b_jev, eff_17b_ctrl)

print(f"1.7B Latency (Ctrl vs Jev): t = {t_lat_17:.3f}, p = {p_lat_17:.4e} (Significant: {p_lat_17 < 0.05})")
print(f"7B Latency (Ctrl vs Jev):   t = {t_lat_7:.3f}, p = {p_lat_7:.4e} (Significant: {p_lat_7 < 0.05})")
print(f"1.7B Recall (Jev vs Regex): t = {t_rec_17:.3f}, p = {p_rec_17:.4e} (Significant: {p_rec_17 < 0.05})")
print(f"1.7B Token Eff (Jev vs Ctrl): t = {t_eff_17:.3f}, p = {p_eff_17:.4e} (Significant: {p_eff_17 < 0.05})")
print(f"7B Token Eff (Jev vs Ctrl):   t = {t_eff_7:.3f}, p = {p_eff_7:.4e} (Significant: {p_eff_7 < 0.05})")

# 95% Confidence Intervals
def ci95(data):
    m = np.mean(data)
    se = stats.sem(data)
    return m, m - 1.96*se, m + 1.96*se

print("\n95% Confidence Intervals:")
print("1.7B Ctrl Latency:  {:.1f} [{:.1f}, {:.1f}]".format(*ci95(lat_17b_ctrl)))
print("1.7B Jev Latency:   {:.1f} [{:.1f}, {:.1f}]".format(*ci95(lat_17b_jev)))
print("7B Ctrl Latency:    {:.1f} [{:.1f}, {:.1f}]".format(*ci95(lat_7b_ctrl)))
print("7B Jev Latency:     {:.1f} [{:.1f}, {:.1f}]".format(*ci95(lat_7b_jev)))
print("1.7B Regex Recall:  {:.1f}% [{:.1f}%, {:.1f}%]".format(*ci95(rec_17b_regex)))
print("1.7B Jev Recall:    {:.1f}% [{:.1f}%, {:.1f}%]".format(*ci95(rec_17b_jev)))
print("7B Ctrl Efficiency: {:.3f} [{:.3f}, {:.3f}]".format(*ci95(eff_7b_ctrl)))
print("7B Jev Efficiency:  {:.3f} [{:.3f}, {:.3f}]".format(*ci95(eff_7b_jev)))
