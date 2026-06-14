# Window Size Experiment

## Recommendation

- Recommended window: **3.0 sec**
- Hop size: **1.5 sec**
- Selection score: **0.532**

Selection score combines:
- Macro-F1 (40%)
- Emotion Match Rate (25%)
- Top-K Accuracy (20%)
- Mean Top-1 Music Match (15%)

## Summary Table

| window_size | hop_size | run_id | n_results | n_groups | mean_music_match_score | best_music_match_score | mean_top1_music_match | emotion_match_rate | macro_f1 | top_k_accuracy | group_consistency_mean | mean_eeg_windows_per_trial | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3.000 | 1.500 | window_size_vkr_w3_h1p5 | 55 | 9 | 0.775 | 0.837 | 0.807 | 0.418 | 0.323 | 0.889 | 0.556 | 3.909 | 0.532 |
| 4.000 | 2.000 | window_size_vkr_w4_h2 | 140 | 10 | 0.766 | 0.855 | 0.823 | 0.200 | 0.196 | 0.900 | 0.500 | 3.000 | 0.432 |
| 12.000 | 6.000 | window_size_vkr_w12_h6 | 150 | 10 | 0.735 | 0.806 | 0.781 | 0.190 | 0.193 | 0.800 | 0.500 | 4.133 | 0.402 |
| 8.000 | 4.000 | window_size_vkr_w8_h4 | 150 | 10 | 0.746 | 0.815 | 0.791 | 0.160 | 0.156 | 0.800 | 0.583 | 4.267 | 0.381 |
| 6.000 | 3.000 | window_size_vkr_w6_h3 | 150 | 10 | 0.760 | 0.859 | 0.824 | 0.140 | 0.132 | 0.600 | 0.500 | 3.900 | 0.331 |
| 16.000 | 8.000 | window_size_vkr_w16_h8 | 150 | 10 | 0.716 | 0.806 | 0.770 | 0.130 | 0.119 | 0.500 | 0.500 | 4.533 | 0.295 |
| 2.000 | 1.000 | window_size_vkr_w2_h1 | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## How to use in thesis

- Compare 4 s and 8 s on the same participant/trial subset.
- Use Macro-F1 and Emotion Match Rate as primary evidence.
- Use Mean Top-1 Music Match and Avg EEG windows per trial as supporting evidence.
- If 8 s gives higher emotion metrics with stable music match, justify 8 s as the main operating window.
