# HLS AI Architect: Universal Hardware Scheduler

A high-performance Reinforcement Learning environment and GNN-based agent designed to automatically schedule Python Data Dependency Graphs (DAGs) onto constrained silicon architectures.

This project was built for the **Meta x Scaler OpenEnv Hackathon** and is optimized for the **NVIDIA Nemotron-3 Super** reasoning model.

## 🚀 Key Features

- **Universal Gauntlet Curriculum**: Trained on a suite of 10 high-complexity hardware stressors (FFT Butterfly, Sobel Stencil, Separable Conv) to ensure zero performance variance on unseen kernels.
- **GNN-Powered Reasoning**: Uses a Graph Neural Network (GNN) features extractor to understand topological bottlenecks and critical paths directly from AST-compiled silicon.
- **"Frontier Sensing" Observation**: A 5-dimensional global state vector that provides the agent with "eyes" on resource congestion and ready-node availability.
- **8GB RAM Optimized**: Designed with segmented replay buffers and frequent gradient flushing to maintain a lightweight memory footprint in Hugging Face environments.

## 🛠️ Environment Specification

### Two-Phase Execution
1. **TRANSFORM (Phase 3 Architect)**: The agent applies HLS Pragmas (`UNROLL`, `PIPELINE`, `ARRAY_PARTITION`) to physically mutate the graph and memory controllers within a strict Area budget.
2. **SCHEDULE (Phase 1 & 2)**: The agent performs cycle-by-cycle scheduling of operations onto physical ALUs, MAC units, and MEM ports (4 ALU, 2 MAC, 2 MEM).

### Observation Space
- **Node Features (300x6)**: OpCode, State, Latency, In-Degree, Criticality, and Applied Pragma.
- **Adjacency Matrix (300x300)**: Full topological dependency map.
- **Global State (5-dim)**: `[Cycle, Busy_Resources, Progress_%, Area_%, Ready_Node_Count]`.

### Action Space (1,801 Actions)
- Actions `0` to `1799`: (Node_ID * 6) + Transformation_Type.
- Action `1800`: Global `WAIT` / `COMMIT` button.

## 📊 Gauntlet Performance Benchmarks

The following results were verified on the final trained model (`ultimate_gauntlet_agent.zip`) under Phase 2/3 hardware constraints (4 ALU, 2 MAC, 2 MEM).

| Kernel Name | Cycles (MakeSpan) | Nodes | Status |
| :--- | :--- | :--- | :--- |
| **gauntlet_fft_butterfly.py** | 13 | 27 | SUCCESS |
| **gauntlet_sobel_stencil.py** | 15 | 50 | SUCCESS |
| **gauntlet_matmul_2x2.py** | 21 | 46 | SUCCESS |
| **gauntlet_separable_conv.py** | 16 | 20 | SUCCESS |
| **gauntlet_iir_serial.py** | 26 | 23 | SUCCESS |
| **gauntlet_conv2d.py** | 20 | 279 | SUCCESS |
| **gauntlet_trap_dynamic.py** | -- | 2 | **REJECTED (Safe)** |
| **kernel.py (4x4 MatVec)** | 67 | 222 | SUCCESS |

*Note: The system correctly identifies and rejects "Trap" kernels containing non-synthesizable dynamic logic (while loops/recursion) to protect the hardware backend.*

## 🛠️ Setup & Installation

1. **Clone & Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Hardening (The Victory Lap)**:
   ```bash
   python victory_lap.py
   ```
   This runs the agent through the full 10-kernel Gauntlet Stressor suite and prints the final cycle counts.

3. **Launch the Dashboard**:
   ```bash
   python server/app.py
   ```
   Access the **Judge Magnet UI** to upload custom kernels and view real-time Gantt chart scheduling.

## 🤖 Inference (Nemotron-Ready)

The project includes a compliant `inference.py` for automated evaluation:
```bash
export HF_TOKEN="your_token"
python inference.py
```
This script uses the `OpenEnvStringWrapper` to bridge the gap between LLM reasoning (e.g., `"SCHEDULE 5"`) and discrete hardware execution.

---
**Submission Metadata:**
- **Target Model**: `nvidia/nemotron-3-8b-super`
- **Benchmark**: `ultimate_gauntlet_v1`
- **Memory Limit**: < 8GB