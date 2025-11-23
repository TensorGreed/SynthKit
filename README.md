# SynthKit

SynthKit is a batteries-included toolkit for harvesting raw documents, generating synthetic QA/CoT pairs, curating them with LLM judges, and exporting the curated corpus into training-ready formats.

## Why SynthKit?

- **End-to-end workflow**: Harvest, mint, audit, and package stages are wired together with a single CLI.
- **Provider flexibility**: Works with OpenAI, Anthropic, Ollama, or any OpenAI-compatible HTTP backend.
- **Extensible architecture**: Register your own generators and exporters without forking the core.
- **Production-conscious**: Typed configuration, retry-enabled HTTP clients, and robust logging paths.

## Installation

```bash
git clone https://github.com/<your-org>/SynthKit.git
cd SynthKit
pip install -e .[test]  # includes dev dependencies like pytest
```

Python 3.10+ is required.

## Configuration

Project settings live under `config/project.yaml`. The schema is defined in `synthkit/config.py` and includes:

- `io`: Input + working directory paths.
- `models`: Logical references to the LLMs used per stage.
- `prompts`: Templates for QA/COT generation and rating.
- `generation`/`curation`: Tunable hyperparameters (chunking size, min judge score, etc.).
- `providers`: Provider definitions (`type` = `openai` | `anthropic` | `http` | `ollama`, API base, auth settings, etc.).

Copy `config/project.example.yaml` to `config/project.yaml` and customize paths, models, and prompts for your environment.

## Running the Pipeline

SynthKit ships with a Typer-powered CLI:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...

# Inspect configuration
python -m synthkit.cli system-check

# Run individual stages
python -m synthkit.cli harvest
python -m synthkit.cli mint --kind qa
python -m synthkit.cli audit
python -m synthkit.cli package --fmt alpaca

# Or execute the entire workflow
python -m synthkit.cli run-all --kind qa --fmt alpaca
```

Each command respects the `--config` flag for alternative configs and `--log-level` for logging verbosity.

### Using Ollama / Open Models

1. Install [Ollama](https://ollama.com/) and run `ollama serve` locally (default `http://localhost:11434`).
2. Pull your desired model (e.g., `ollama pull mistral`).
3. Configure a provider entry:

   ```yaml
   providers:
     local-ollama:
       type: ollama
       api_base: http://localhost:11434
   models:
     mint_generator:
       provider: local-ollama
       name: mistral
   ```

No API key is required; SynthKit will call `/api/chat` with non-streaming requests and respect `temperature`/`max_tokens` from the config. Use `--kind qa` (or your custom generator) exactly as with hosted providers.

### Using vllm / Open Models
I have used AMD Instinct MI300X GPU droplet on DigitalOcean for my testing. Feel free to modify the steps for the GPU of your choice.
```
apt update && apt upgrade -y

apt install -y ca-certificates curl gnupg

install -m 0755 -d /etc/apt/keyrings

curl -fsSL https://download.docker.com/linux/ubuntu/gpg   | gpg --dearmor -o /etc/apt/keyrings/docker.gpg

chmod a+r /etc/apt/keyrings/docker.gpg

echo   "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu \
$(lsb_release -cs) stable"   > /etc/apt/sources.list.d/docker.list

apt install -y docker-ce docker-ce-cli containerd.io

docker pull rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4

export MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct  # Model of your choice
export DOCKER_IMG=rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4
export HOST_PORT=8000
export HF_TOKEN="<HUGGINGFACE_TOKEN>"
export MODEL=mistralai/Mistral-7B-Instruct-v0.3  # Model of your choice

docker run --rm  \   
--device=/dev/kfd --device=/dev/dri --group-add video  \  
--shm-size 16G  \  
-p $HOST_PORT:8000  \  
--security-opt seccomp=unconfined  \
--security-opt apparmor=unconfined  \  
--cap-add=SYS_PTRACE  \  
-v $(pwd):/workspace  \  
--env HUGGINGFACE_HUB_CACHE=/workspace  \    
--env VLLM_USE_TRITON_FLASH_ATTN=0  \ 
--env PYTORCH_TUNABLEOP_ENABLED=1  \ 
--env HF_TOKEN=$HF_TOKEN  \ 
$DOCKER_IMG python3 -m vllm.entrypoints.openai.api_server  \ 
--model $MODEL  \ 
--host 0.0.0.0  \ 
--port 8000  \ 
--dtype float16  \ 
--gpu-memory-utilization 0.9  \ 
--max-model-len 8192  \ 
--max-num-batched-tokens 32768  \ 
--swap-space 16  \ 
--disable-log-requests  \      
--api-key my-secret-key
```

## Extending SynthKit

The `synthkit.extensions` module exposes registries for generator factories and export formatters:

```python
from synthkit.extensions import register_generator, register_formatter

# Register a custom generator
register_generator("dialog", lambda client, cfg: MyDialogGenerator(client, cfg))

# Register a new formatter
register_formatter("json-min", lambda sample: {...})
```

- Generators must implement the `BaseGenerator` interface (`build_messages` + `parse_output`).
- Formatters accept a curated sample dictionary and return a transformed payload.
- Once registered, custom names become available in the CLI (`--kind dialog`, `--fmt json-min`).

You can ship third-party extensions by importing them in your project before invoking the CLI (e.g., via a bootstrap script or inside `config/__init__.py`).

## Testing

Unit tests cover HTTP client behavior, router caching, and extension registries:

```bash
pytest
```

Use `pip install -e .[test]` to pull in test dependencies.

## Contributing

1. Fork and clone the repository.
2. Create a feature branch (`git checkout -b feature/my-change`).
3. Add tests and ensure `pytest` passes.
4. Submit a pull request describing the change and testing performed.

Feel free to open issues for bugs, feature requests, or extension proposals.

## License

[License](/LICENSE)
