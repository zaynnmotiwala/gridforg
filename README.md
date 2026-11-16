# GridForge: Distributed GPU Worker for AI Image Generation

GridForge turns your GPU into a node on the **AI Horde**, a free, crowdsourced network for AI image generation. The worker fetches image-generation jobs from the network, runs them locally with Stable Diffusion, SDXL, Stable Cascade, or Flux models, runs safety checks on the results, and uploads them. In return you earn **kudos**, which give your own requests on the network higher priority.

---

## Table of Contents

- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Recommended Settings by GPU](#recommended-settings-by-gpu)
- [Running the Worker](#running-the-worker)
- [Updating](#updating)
- [Custom Models](#custom-models)
- [Docker](#docker)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Key Features

- **Multi-process architecture.** A process manager coordinates separate inference and safety processes, so one failure never brings down the whole worker.
- **Broad model support.** Works with SD 1.5, SD 2.x, SDXL, Stable Cascade, and Flux (schnell), plus LoRAs, ControlNet, and post-processing.
- **Built-in safety checks.** A dedicated safety process scans every generated image before upload, on either the GPU or the CPU.
- **Job batching and queueing.** Configurable queue size, thread count, and batch limits get the most throughput out of your hardware.
- **Smart VRAM management.** High-, moderate-, and low-memory modes, optional model unloading, and preload timeouts keep models running on 8 GB to 48 GB+ GPUs.
- **Cross-platform.** Runs on Windows and Linux with NVIDIA CUDA, plus experimental AMD ROCm and DirectML support.
- **Custom models.** Serve your own `.safetensors` checkpoints (requires approval on the network).
- **Self-contained runtime.** The setup scripts install an isolated Python environment through micromamba, so there's no system-wide setup.
- **Docker support** for CUDA and ROCm, with Compose files.
- **Detailed logging.** Each process writes its own logs, plus separate trace logs for errors.

---

## How It Works

```text
                 ┌──────────────────────────────────────────────┐
                 │                 Process Manager              │
 AI Horde API ◄──┤  pops jobs · tracks kudos · manages queue    │
   (jobs)        │  loads models · submits results              │
                 └───────┬──────────────────────┬───────────────┘
                         │                      │
                ┌────────▼────────┐    ┌────────▼────────┐
                │   Inference     │    │     Safety      │
                │   process(es)   │───►│     process     │
                │  (horde-engine) │    │ (horde-safety)  │
                └─────────────────┘    └────────┬────────┘
                                                │ approved image
                                                ▼
                                       Upload to AI Horde → kudos
```

1. The **process manager** asks the AI Horde API for jobs that match your configured models and limits.
2. An **inference process** loads the requested model and generates the image.
3. The **safety process** checks the output before it leaves your machine.
4. The result is **submitted** to the network, and your worker earns kudos.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep learning | PyTorch 2.9 (CUDA, ROCm, DirectML) |
| Inference engine | `horde-engine` (ComfyUI-based pipelines) |
| API client | `horde-sdk` |
| Safety | `horde-safety` |
| Model catalog | `horde-model-reference` |
| Config and validation | Pydantic, ruamel.yaml, python-dotenv |
| Runtime | micromamba (isolated environment) |
| Containers | Docker, Docker Compose |
| Tooling | pytest, mypy, Ruff, Black, pre-commit |

---

## Requirements

| | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA with 8 GB VRAM | 12–24 GB+ VRAM |
| System RAM | 16 GB | 32 GB+ (48 GB+ for Flux or Cascade) |
| Storage | SSD | NVMe SSD with plenty of free space for models |
| Swap | 8 GB | 16 GB+ |
| OS | Windows 10/11 or Linux | Linux for AMD |

You also need a free **AI Horde account and API key**. Register at [aihorde.net/register](https://aihorde.net/register) and keep the key private, like a password.

---

## Installation

> **Note:** Don't use spaces in the installation path. `C:\gridforge` works; `C:\My Workers` does not.

### Windows

```cmd
cd C:\path\to\install\folder
git clone https://github.com/zaynnmotiwala/gridforge.git
cd gridforge
```

No Git? Download the repository as a ZIP from GitHub and extract it.

### Linux

```bash
git clone https://github.com/zaynnmotiwala/gridforge.git
cd gridforge
```

### AMD GPUs (experimental, Linux only)

Use `horde-bridge-rocm.sh` and `update-runtime-rocm.sh` instead of the standard scripts. See [README_advanced.md](README_advanced.md) for ROCm and WSL setup.

### DirectML (experimental)

Use `update-runtime-directml.cmd` and `horde-bridge-directml.cmd`. DirectML works but is much slower than CUDA or ROCm. See [README_advanced.md](README_advanced.md).

---

## Configuration

1. Copy the template:

   ```bash
   cp bridgeData_template.yaml bridgeData.yaml      # Windows: copy bridgeData_template.yaml bridgeData.yaml
   ```

2. Open `bridgeData.yaml` and set at least these values:

   | Setting | Description |
   |---|---|
   | `api_key` | Your AI Horde API key |
   | `dreamer_name` | A unique name for your worker across the whole network (a taken name causes a "Wrong Credentials" error) |
   | `models_to_load` | The models your worker will offer |
   | `max_power` | The maximum image size and steps your worker accepts |
   | `max_threads` | How many jobs run in parallel |
   | `queue_size` | How many jobs to prefetch (uses system RAM) |

The template documents every other option in detail.

> **Never commit `bridgeData.yaml`.** It contains your API key.

---

## Recommended Settings by GPU

**24 GB+ VRAM** (RTX 3090 / 4090)

```yaml
queue_size: 1                 # <32GB RAM: 0 · 32GB: 1 · >32GB: 2
safety_on_gpu: true
high_memory_mode: true
high_performance_mode: true
unload_models_from_vram_often: false
max_threads: 1                # 2 is often viable on xx90 cards
post_process_job_overlap: true
max_power: 64                 # lower if max_threads: 2
max_batch: 8
allow_sdxl_controlnet: true
```

**12–16 GB VRAM** (RTX 3080 Ti / 4070 Ti / 4080)

```yaml
queue_size: 1
safety_on_gpu: true           # consider false with Cascade or Flux
moderate_performance_mode: true
unload_models_from_vram_often: false
max_threads: 1
max_power: 50
max_batch: 4
```

**8–10 GB VRAM** (RTX 2080 / 3060 / 4060)

```yaml
queue_size: 1
safety_on_gpu: false
max_threads: 1
max_power: 32                 # don't go higher
max_batch: 4                  # don't go higher
allow_post_processing: false  # can be true if you skip SDXL and Flux
allow_sdxl_controlnet: false
```

**Less than 32 GB of system RAM:** use `queue_size: 0` and `load_large_models: false`, and add `ALL SDXL` and `ALL SD21` to `models_to_skip`.

**Slower GPUs:** `limit_max_steps: true` caps the steps per job, `preload_timeout: 120` allows slower model loads, and `extra_slow_worker: true` gives each job more time (users must opt in to slow workers).

### Performance tips

- Keep models on an **SSD**. On an HDD, offer only one model.
- Keep `max_threads` at **2 or below** unless you have a 48 GB+ data-center GPU.
- SDXL needs about **9 GB of free RAM**. Flux and Stable Cascade need about **20 GB**.
- Turn off sleep and power-saving modes while the worker runs.

---

## Running the Worker

| Action | Windows | Linux |
|---|---|---|
| Start | `horde-bridge.cmd` | `./horde-bridge.sh` |
| Start (AMD) | n/a | `./horde-bridge-rocm.sh` |
| Stop | `Ctrl+C` (current jobs finish first) | `Ctrl+C` |

The first run installs the runtime and downloads your models, which can take a while.

### Monitoring

The terminal shows live progress, completed jobs, kudos earned, and errors. Full logs are written to `logs/`:

| File | Contents |
|---|---|
| `bridge.log` | All activity from the main process |
| `bridge_<n>.log` | Activity from process `n` |
| `trace.log`, `trace_<n>.log` | Errors and warnings only |

### Multiple GPUs

Run one worker instance per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 ./horde-bridge.sh -n "Instance 1"
CUDA_VISIBLE_DEVICES=1 ./horde-bridge.sh -n "Instance 2"
```

Each instance needs its own share of system RAM, so plan for 32–64 GB or more.

---

## Updating

1. Stop the worker with `Ctrl+C`.
2. Pull the latest code:

   ```bash
   git pull
   ```

3. Update the runtime and dependencies:

   | OS / GPU | Script |
   |---|---|
   | Windows (NVIDIA) | `update-runtime.cmd` |
   | Linux (NVIDIA) | `./update-runtime.sh` |
   | Linux (AMD) | `./update-runtime-rocm.sh` |
   | Windows (DirectML) | `update-runtime-directml.cmd` |

4. Start the worker again.

> Some antivirus tools (such as Avast) can block the update. If you see `CRYPT_E_NO_REVOCATION_CHECK`, pause the antivirus, run the update, then turn it back on.

---

## Custom Models

Serving models that aren't in the official catalog requires the **customizer** role on the AI Horde. Once you have it, add the model to `bridgeData.yaml`:

```yaml
custom_models:
  - name: My Custom Model
    baseline: stable_diffusion_xl
    filepath: /path/to/model.safetensors

models_to_load:
  - My Custom Model
```

**Supported baselines:** `stable_diffusion_1`, `stable_diffusion_2_768`, `stable_diffusion_2_512`, `stable_diffusion_xl`, `stable_cascade`, `flux_1`

Some rules:
- Only **Flux.schnell**-based models are allowed. Flux.dev and its derivatives are not permitted.
- Custom model names can't match existing catalog names, and they can't be explicit or vulgar.
- Custom models are treated as SD 1.5 for kudos and safety checks.
- When setup works, `custom_models.json` appears in the worker folder on startup.

---

## Docker

Build and run with the included Dockerfiles and Compose files for **CUDA** and **ROCm**:

```bash
cd Dockerfiles
docker compose -f compose.cuda.yaml up -d --build      # NVIDIA
docker compose -f compose.rocm.yaml up -d --build      # AMD
```

See [Dockerfiles/README.md](Dockerfiles/README.md) for build arguments, volumes, and GPU passthrough.

---

## Project Structure

```text
├── horde_worker_regen/
│   ├── process_management/
│   │   ├── process_manager.py     # Core orchestrator: job pop, queue, submit, kudos
│   │   ├── inference_process.py   # Image generation
│   │   ├── safety_process.py      # Output safety checks
│   │   ├── horde_process.py       # Shared base class for child processes
│   │   └── messages.py            # Inter-process message types
│   ├── bridge_data/               # Config loading and validation
│   ├── download_models.py         # Model downloader
│   ├── load_env_vars.py           # Environment-variable config
│   └── run_worker.py              # Entry point
├── Dockerfiles/                   # CUDA and ROCm images and Compose files
├── tests/                         # pytest suite
├── bridgeData_template.yaml       # Fully documented config template
├── horde-bridge.* / update-runtime.*   # Start and update scripts (cmd and sh)
└── README_advanced.md             # Manual setup, ROCm, WSL, DirectML
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| **Model download fails** | Check your free disk space and internet connection. |
| **Jobs time out** | Remove large models (Flux, Cascade, SDXL), lower `max_power`, or turn off `allow_post_processing`, `allow_controlnet`, `allow_sdxl_controlnet`, or `allow_lora`. |
| **Out of memory** | Lower `max_threads`, `max_batch`, or `queue_size`, and close other GPU-heavy apps. |
| **"Wrong Credentials" error** | Your `dreamer_name` is already taken. Choose another name. |
| **Fewer kudos than expected** | New workers have part of their kudos held for about a week until they become trusted. After that, you get the held kudos and full rewards. |
| **Worker is in maintenance mode** | The server pauses workers that are too slow or fail too many jobs. Search `logs/` for `ERROR`, fix the cause, then unpause the worker from your AI Horde account settings. |

To report a bug or request a feature, open an issue on this repository.

---

## License

This project is licensed under the **GNU Affero General Public License v3.0**. See [LICENSE](LICENSE).

It is a modified version of the open-source AI Horde reGen image worker, redistributed under the same license.

Many supported models use the **CreativeML OpenRAIL** license. Check each model's license before you use it.
