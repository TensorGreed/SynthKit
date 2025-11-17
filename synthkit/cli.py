from __future__ import annotations
import typer

from .config import load_config
from .logging_config import configure_logging
from .pipeline.harvest import run_harvest
from .pipeline.mint import run_mint
from .pipeline.audit import run_audit
from .pipeline.package import run_package
from .pipeline.run_all import run_pipeline

app = typer.Typer(help="SynthForge – synthetic data generation & curation toolkit")


@app.callback()
def main(
    ctx: typer.Context,
    config: str = typer.Option("config/project.example.yaml", "--config", "-c"),
    log_level: str = typer.Option("INFO", "--log-level"),
):
    configure_logging(log_level)
    ctx.obj = load_config(config)


@app.command()
def system_check(ctx: typer.Context):
    """Basic sanity checks."""
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
    kind: str = typer.Option("qa", "--kind", help="qa | cot"),
):
    """Generate synthetic data."""
    cfg = ctx.obj
    out = run_mint(cfg, generator_type=kind)  # type: ignore[arg-type]
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
    fmt: str = typer.Option("alpaca", "--fmt", help="alpaca | chatml | openai-ft"),
):
    """Export curated data into final training formats."""
    cfg = ctx.obj
    out = run_package(cfg, fmt=fmt)  # type: ignore[arg-type]
    typer.echo(f"Packaged {len(out)} datasets.")


@app.command(name="run-all")
def run_all(
    ctx: typer.Context,
    kind: str = typer.Option("qa", "--kind"),
    fmt: str = typer.Option("alpaca", "--fmt"),
):
    """Run full pipeline: harvest → mint → audit → package."""
    cfg = ctx.obj
    run_pipeline(cfg, generator_type=kind, export_fmt=fmt)  # type: ignore[arg-type]
    typer.echo("Pipeline completed.")
