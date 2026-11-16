# SmolTalkBench results

100 prompts. Judge: Qwen3.6-35B-A3B Q4 (local), AlpacaEval 2.0 weighted prompt, WildBench length penalty. Every row was scored by this same judge.

| System | Win rate | 95% CI | Turns sent to a model |
|---|---:|---|---:|
| PyBot (code only) | 17.3% | 10.8–24.4 | 0% |
| SmolLM2-360M (brief prompt) | 38.8% | 30.9–46.6 | 100% |

## PyBot → smollm2:360m

Model alone: 38.8%. Picking the better of the two per prompt with hindsight: 44.8%.

| Target model share | Actual | Win rate | 95% CI | Gain over PyBot | Random routing, same calls | Oracle, same calls |
|---:|---:|---:|---|---:|---:|---:|
| 1% | 3.0% | 17.5% | 10.8–24.5 | +0.2 (+0.0 to +0.5) | 18.0% | 20.3% |
| 2% | 3.0% | 17.5% | 10.8–24.5 | +0.2 (+0.0 to +0.5) | 18.0% | 20.3% |
| 5% | 7.0% | 18.3% | 11.7–25.5 | +1.0 (+0.0 to +2.8) | 18.8% | 24.3% |
| 10% | 14.0% | 21.3% | 14.5–28.7 | +4.0 (+1.0 to +7.6) | 20.3% | 30.6% |
| 15% | 17.0% | 20.7% | 14.1–27.9 | +3.3 (+0.0 to +7.1) | 21.0% | 32.8% |
| 20% | 23.0% | 22.9% | 16.1–30.6 | +5.6 (+1.6 to +10.3) | 22.2% | 36.7% |
| 25% | 30.0% | 24.8% | 17.8–32.5 | +7.5 (+3.1 to +12.4) | 23.7% | 40.2% |
| 30% | 36.0% | 26.9% | 19.7–34.9 | +9.6 (+4.3 to +15.3) | 25.0% | 42.3% |
| 40% | 46.0% | 31.9% | 24.4–39.7 | +14.6 (+8.6 to +21.3) | 27.2% | 44.1% |
| 50% | 53.0% | 34.6% | 26.9–42.5 | +17.3 (+11.2 to +24.0) | 28.7% | 44.6% |
| 75% | 76.0% | 33.1% | 25.5–40.7 | +15.8 (+8.2 to +23.7) | 33.6% | 44.8% |
| 100% | 100.0% | 38.8% | 30.9–46.6 | +21.5 (+12.5 to +30.1) | 38.8% | 38.8% |
