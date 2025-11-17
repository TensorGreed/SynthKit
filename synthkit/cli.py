"""Typer-powered entry points for the SynthKit pipeline."""

from __future__ import annotations

from typing import Sequence

import typer

from .config import load_config
from .extensions import available_generator_types, available_formatter_names
from .logging_config import configure_logging
from .pipeline.harvest import run_harvest
from .pipeline.mint import run_mint
from .pipeline.audit import run_audit
from .pipeline.package import run_package
from .pipeline.run_all import run_pipeline

app = typer.Typer(help="SynthForge - synthetic data generation & curation toolkit")


def _describe_options(values: Sequence[str]) -> str:
    return ", ".join(values) if values else "none registered"


def _normalize_choice(label: str, value: str, options: Sequence[str]) -> str:
    normalized = value.lower()
    if options and normalized not in options:
        raise typer.BadParameter(
            f"Unknown {label} '{value}'. Available: {_describe_options(options)}"
        )
    return normalized


def _generator_help() -> str:
    return f"Generator kind (registered: {_describe_options(available_generator_types())})."


def _formatter_help() -> str:
    return f"Export format (registered: {_describe_options(available_formatter_names())})."


GENERATOR_DEFAULT = (available_generator_types()[0] if available_generator_types() else "qa")
FORMATTER_DEFAULT = (available_formatter_names()[0] if available_formatter_names() else "alpaca")


@app.callback()
def main(
    ctx: typer.Context,
    config: str = typer.Option("config/project.example.yaml", "--config", "-c"),
    log_level: str = typer.Option("INFO", "--log-level"),
):
    """Initialize logging and load the project config before running a command."""
    configure_logging(log_level)
    ctx.obj = load_config(config)


@app.command()
def system_check(ctx: typer.Context):
    """Print a summary of the resolved configuration for quick validation."""
    cfg = ctx.obj
    typer.echo(f"Loaded config with input root: {cfg.io.input_root}")
    typer.echo(f"Working root: {cfg.io.working_root}")
    typer.echo(f"Providers: {', '.join(cfg.providers.keys())}")


@app.command()
def harvest(ctx: typer.Context):
    """Ingest and normalize raw documents."""
    cfg = ctx.obj
    out = run_harvest(cfg)
    typer.echo(f"Harvested {len(out)} documents.")


@app.command()
def mint(
    ctx: typer.Context,
    kind: str = typer.Option(GENERATOR_DEFAULT, "--kind", help=_generator_help()),
):
    """Generate synthetic data from harvested documents."""
    cfg = ctx.obj
    normalized_kind = _normalize_choice("generator kind", kind, available_generator_types())
    out = run_mint(cfg, generator_type=normalized_kind)
    typer.echo(f"Minted synthetic data into {len(out)} files.")


@app.command()
def audit(ctx: typer.Context):
    """Curate synthetic data using LLM-as-judge."""
    cfg = ctx.obj
    out = run_audit(cfg)
    typer.echo(f"Audited {len(out)} files.")


@app.command(name="package")
def package_cmd(
    ctx: typer.Context,
    fmt: str = typer.Option(FORMATTER_DEFAULT, "--fmt", help=_formatter_help()),
):
    """Export curated data into final training formats."""
    cfg = ctx.obj
    normalized_fmt = _normalize_choice("format", fmt, available_formatter_names())
    out = run_package(cfg, fmt=normalized_fmt)
    typer.echo(f"Packaged {len(out)} datasets.")


@app.command(name="run-all")
def run_all(
    ctx: typer.Context,
    kind: str = typer.Option(GENERATOR_DEFAULT, "--kind", help=_generator_help()),
    fmt: str = typer.Option(FORMATTER_DEFAULT, "--fmt", help=_formatter_help()),
):
    """Run the full pipeline end-to-end: harvest -> mint -> audit -> package."""
    cfg = ctx.obj
    normalized_kind = _normalize_choice("generator kind", kind, available_generator_types())
    normalized_fmt = _normalize_choice("format", fmt, available_formatter_names())
    run_pipeline(cfg, generator_type=normalized_kind, export_fmt=normalized_fmt)
    typer.echo("Pipeline completed.")
