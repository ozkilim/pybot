# SmolTalkBench results

100 prompts. Judge: Qwen3.6-35B-A3B Q4 (local), AlpacaEval 2.0 weighted prompt, WildBench length penalty. Every row was scored by this same judge.

| System | Win rate | 95% CI | Turns sent to a model |
|---|---:|---|---:|
| PyBot (code only) | 17.3% | 10.8–24.4 | 0% |
| SmolLM2-360M | 35.2% | 28.4–41.9 | 100% |
| SmolLM2-360M (brief prompt) | 38.8% | 30.9–46.6 | 100% |
| SmolLM2-1.7B (brief prompt) | 55.3% | 47.3–62.9 | 100% |
| Gemma3-1B (brief prompt) | 50.8% | 42.7–58.9 | 100% |

## PyBot → smollm2:360m

Model alone: 38.8%. Picking the better of the two per prompt with hindsight: 44.8%.

| Target model share | Actual | Win rate | 95% CI | Gain over PyBot | Random routing, same calls | Oracle, same calls |
|---:|---:|---:|---|---:|---:|---:|
| 1% | 2.0% | 17.6% | 11.0–24.5 | +0.3 (+0.0 to +0.8) | 17.7% | 19.3% |
| 2% | 2.0% | 17.6% | 11.0–24.5 | +0.3 (+0.0 to +0.8) | 17.7% | 19.3% |
| 5% | 5.0% | 17.7% | 11.2–24.6 | +0.4 (+0.0 to +1.1) | 18.4% | 22.3% |
| 10% | 13.0% | 21.0% | 14.3–28.1 | +3.7 (+1.0 to +7.3) | 20.1% | 29.8% |
| 15% | 21.0% | 23.2% | 16.2–30.6 | +5.9 (+2.6 to +9.8) | 21.8% | 35.7% |
| 20% | 29.0% | 25.3% | 18.2–32.8 | +8.0 (+4.1 to +12.7) | 23.5% | 39.7% |
| 25% | 31.0% | 26.0% | 18.9–33.5 | +8.7 (+4.5 to +13.4) | 24.0% | 40.7% |
| 30% | 34.0% | 27.2% | 19.9–34.6 | +9.8 (+5.5 to +15.0) | 24.6% | 41.8% |
| 40% | 46.0% | 32.5% | 25.1–40.3 | +15.2 (+9.1 to +21.4) | 27.2% | 44.1% |
| 50% | 52.0% | 34.2% | 26.7–42.0 | +16.9 (+10.5 to +23.7) | 28.5% | 44.5% |
| 75% | 75.0% | 36.4% | 28.7–44.0 | +19.0 (+11.7 to +26.4) | 33.4% | 44.8% |
| 100% | 100.0% | 38.8% | 30.9–46.6 | +21.5 (+12.5 to +30.1) | 38.8% | 38.8% |
