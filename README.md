# aha-moments-during-in-context-learning
This repository contains the code and experimental data for our investigation of discontinuous performance jumps ( "aha moments" ) in large language model in-context learning. We study when and why LLMs suddenly transition from random guessing to competent performance as the number of in-context examples increases.

Key findings:

Emergence is task-dependent and selective, not universal

Specific attention heads (Layer 20 Head 1 in Qwen2-1.5B) develop structured attention patterns precisely at the emergence point

Entropy analysis reveals three regimes: success, uncertain failure, and "confident wrongness"
