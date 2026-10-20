# In-Context Operator Networks for Solving PDEs

A compact reproduction of **ICON-LM** (In-Context Operator Networks with Language captions) applied to the 1D viscous Burgers equation. A 2.21M-parameter transformer learns to solve a PDE from a few example input/output pairs in its prompt, with no weight updates or fine-tuning at inference time.

Built for UCLA MATH 270A by Dhruv Patel and Aditya Patil, based on:

> Yang, L., Liu, S., Osher, S. J. *Fine-Tune Language Models as Multi-Modal Differential Equation Solvers.* [Paper](https://ww3.math.ucla.edu/wp-content/uploads/2024/05/ICON_LM_202405-1.pdf)

## The problem

$$u_t + u\,u_x = \nu\,u_{xx}$$

The hidden operator is the viscosity ν. Each prompt contains J demonstration trajectories (initial condition → solution) generated with the same ν, plus one query initial condition. The model predicts the query's solution by inferring ν from the demonstrations in context.

Prompts can optionally include a natural-language caption describing ν, either **precise** (a numeric value) or **vague** (a qualitative description), to test how much language conditioning helps.

## Results

| Prompt | 0 shots | 1 shot | 2 shots | 3 shots |
|---|---|---|---|---|
| No caption | 21.99% | 15.99% | 15.92% | 15.02% |
| Vague caption | 16.07% | 15.24% | 15.37% | 14.37% |
| Precise caption | 18.38% | 15.60% | 15.66% | 14.93% |

*Relative L2 error on held-out viscosities and initial conditions.*

- **In-context learning works:** with no caption, error falls from 22% to 15% as demonstrations go from 0 to 3, with no retraining. Most of the gain comes from the first example, since one trajectory already reveals how strongly the flow is damped.
- **Captions help most when there are no examples:** a vague caption cuts zero-shot error by about 27% relative to no caption. The benefit shrinks as demonstrations are added.
- **Exact numbers in text are hard to use:** the frozen sentence encoder is weak at distinguishing nearby numeric strings, so precise captions with the viscosity value were not reliably more useful than vague ones.

As a separate sanity check, a nearest-centroid classifier on five hand-crafted trajectory features recovers the viscosity bin with 81% accuracy over five bins, confirming that the trajectories carry recoverable information about ν (see `evaluate.py`).

## Implementation

| File | Purpose |
|---|---|
| `data.py` | Generates Burgers trajectories with a spectral (FFT) right-hand side and SciPy's RK45 solver over log-uniform ν, and builds the precise and vague captions |
| `model.py` | `ICONLM`, a small transformer over `(condition, target, query)` token blocks, plus `build_mask`, the custom attention mask that keeps queries from attending to their own targets |
| `train.py` | Samples prompts (J demos + 1 query) per operator, mixes in sentence-embedding captions, and saves `checkpoint.pt` and `test_data.npz` |
| `evaluate.py` | Forward evaluation of error vs. number of shots for each caption type, plus inverse ν estimation |
| `plot.py` | Produces `scaling_curve.png`, `zero_shot_bar.png`, and `scaling_results.csv` |
| `run.sh` | Runs the full pipeline: train → evaluate → plot |

**Stack:** Python, PyTorch, NumPy, SciPy, sentence-transformers, Matplotlib

## Running it

```bash
cd burgers-example
pip install -r requirements.txt

# Full pipeline
./run.sh

# Or skip training and use the included checkpoint
python evaluate.py
python plot.py
```

A pre-trained `checkpoint.pt` and held-out `test_data.npz` are included, so you can reproduce every figure without retraining.

## Write-up

- [`Paper.pdf`](Paper.pdf): full report with methods and results
- [`Presentation.pdf`](Presentation.pdf): slides from the final presentation
