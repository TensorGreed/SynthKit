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