# sft_model_qwen3_8b Training Summary

- Source state: `outputs/sft_model_qwen3_8b/checkpoint-375/trainer_state.json`
- Global step: 375
- Epoch: 3.0
- Logged points: 75
- Final logged tokens: 1426575.0

| metric | first | last | min | max | avg |
|---|---:|---:|---:|---:|---:|
| loss | 2.3489 | 0.6807 | 0.6807 | 2.3489 | 1.08622 |
| grad_norm | 3.82812 | 0.875 | 0.484375 | 3.82812 | 0.916927 |
| learning_rate | 3.33e-05 | 1.87e-09 | 1.87e-09 | 1.00e-04 | 5.00e-05 |
| mean_token_accuracy | 0.707202 | 0.846045 | 0.688713 | 0.864075 | 0.784929 |
| entropy | 0.406349 | 0.589111 | 0.406349 | 1.13831 | 0.787914 |

## Interpretation

- Training-fit metrics show whether SFT optimization ran normally.
- Loss/accuracy curves do not prove safety alignment quality by themselves.
- Final claims should be based on held-out safety and utility evaluation.
