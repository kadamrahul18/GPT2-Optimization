Feature 4 artifacts: BigPurple V100 2-node 8-GPU GPT-2 DeepSpeed runs.
- Throughput claims should use NSYS=0 runs:
  - accum2_300 vs bucket200_300 training_metrics.json
- Bottleneck attribution uses NSYS=1 run:
  - bucket200_nsys80 profile_summary.json + nsys_stats
- Baseline comm-heavy trace: baseline_2026-01-26
