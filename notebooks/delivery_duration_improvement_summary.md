# Delivery Duration Model – Targeted Improvements Summary

## 1. Time drift diagnosis

- **Monthly mean delivery_duration_days**: Varies by month (e.g. 2017 ~11–15 days, 2018-06 to 2018-08 ~8–9 days). Later months (test window) have **shorter** mean delivery times.
- **Train vs test target distribution**:
  - Train: mean=13.82 days, median=11.48, p99=48.9.
  - Test: mean=8.37 days, median=7.23, p99=28.0.
- **Tail frequency**: Train % > 30 days = 6.04%; Test % > 30 days = 0.73%. Tail frequency **decreased** in the test period, so the model (trained on more long deliveries) overpredicts in a test set with fewer long deliveries → negative test R² and systematic overprediction.

*Output*: `notebooks/delivery_duration_time_drift.txt`

---

## 2. Heavy tail handling

- **Winsorization**: Optional `target_winsorize_p99` in config (default false). When true, target is capped at train 99th percentile before log transform.
- **Log vs raw**: Ridge comparison logged in every run:
  - **Ridge (log target)**: test MAE ≈ 4.79.
  - **Ridge (raw target)**: test MAE ≈ 5.97.
- **Conclusion**: Log-transform clearly helps; Ridge with log outperforms Ridge with raw.
- **Metrics by band (test)**:
  - **y_true ≤ 30 days** (most of test): MAE ≈ 4.67, R² ≈ −0.46.
  - **y_true > 30 days** (n=105): MAE ≈ 24.5, R² ≈ −4.6. Model fails on the long-delivery tail.

---

## 3. Geographic and interaction features

- **Added**:
  - `distance_km_bucket`: categorical (d_0_500, d_500_1500, d_1500_plus, d_unknown).
  - `region_to_region`: seller_region + "_" + customer_region.
  - `inter_state_x_bulky_flag`: (1 − intra_state) × bulky_flag.
- **Segment MAE (test)**:
  - **intra_state**: MAE ≈ 3.06 (n=5925).
  - **inter_state**: MAE ≈ 6.02 (n=8546).
  - **long_distance_bucket**: If geo data is present, segment for d_1500_plus is reported; in the current dataset distance_km is often missing so all rows may be d_unknown.

---

## 4. Overfitting reduction

- **Tree**: Stronger regularization — XGBoost max_depth 4–5 (was 6–8), reg_alpha=0.1, reg_lambda=1.0; RF max_depth=6 (was 12).
- **Val–test gap**: MAE gap ≈ +0.23 days, R² gap ≈ −0.50. Slight improvement vs previous run (smaller MAE gap, test R² −0.22 vs −0.24).

---

## 5. Re-evaluation (run 20260227_015242)

| Metric   | Validation | Test   |
|----------|------------|--------|
| MAE      | 4.58       | 4.81   |
| RMSE     | 6.52       | 6.03   |
| R²       | 0.285      | −0.22  |
| MAPE (%) | 41.5       | 41.9   |

**Segment MAE (test)**: intra_state 3.06, inter_state 6.02.

---

## 6. What improved

- Time drift and train/test distribution documented; explains negative test R².
- Log vs raw comparison: log-target Ridge is clearly better (lower test MAE).
- Segment metrics: intra_state vs inter_state and (when available) long-distance bucket.
- Stronger tree regularization and slightly better test MAE/R² and val–test gap.
- New geo/interaction features (region_to_region, distance_km_bucket, inter_state_x_bulky_flag) in the pipeline for when data is available.

---

## 7. What still fails

- **Test R² remains negative** (−0.22): test period has shorter deliveries and fewer long tails than train → systematic overprediction.
- **Long deliveries (y_true > 30 days)**: MAE ≈ 24.5 days, R² ≈ −4.6; model is not reliable on the tail.
- **inter_state**: MAE ~2× intra_state; long-distance and inter-state shipments remain hardest.
- **distance_km_bucket**: In the current dataset, all rows may be d_unknown (missing geo), so long-distance bucket segment is not discriminative.

---

## 8. Production reliability

- **Use with caution.** Recommend:
  - Rely on the model mainly for **short-to-medium** delivery times (e.g. ≤30 days); use segment-specific thresholds or fallbacks for inter_state and long-distance.
  - **Monitor** validation and test metrics over time; consider retraining or temporal recalibration as distribution shifts.
  - Optional **winsorization** (`target_winsorize_p99: true` in config) to reduce impact of extreme tails during training.

---

## Artifacts

- **Time drift**: `notebooks/delivery_duration_time_drift.txt`
- **Config**: `notebooks/delivery_duration_feature_config.json` (includes new numeric/categorical features)
- **Latest run**: `experiments/20260227_015242/` (model, preprocessor, experiment.json, error_analysis.txt, evaluation_metrics.json)
- **Diagnostics script**: `src/diagnose_delivery_duration_drift.py`
