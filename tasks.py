"""Invokes tasks"""
from invoke import task


ENTRYPOINT_PATH = "train.py"


@task
def run(ctx):
    """Runs the application"""
    ctx.run(f"poetry run python3 {ENTRYPOINT_PATH}", echo=True)


@task
def format(ctx):
    """Formats Python code"""
    ctx.run(f"poetry run black {ENTRYPOINT_PATH}", echo=True)
    ctx.run(f"poetry run isort {ENTRYPOINT_PATH}", echo=True)


@task
def lint(ctx):
    """Lints Python code"""
    ctx.run(f"poetry run flake8 --show-source {ENTRYPOINT_PATH}", echo=True)
    ctx.run(f"poetry run pylint {ENTRYPOINT_PATH}", echo=True)


@task
def type_check(ctx):
    """Checks types of our Python source code"""
    ctx.run(f"poetry run mypy {ENTRYPOINT_PATH}", echo=True)


@task(pre=[format, lint, type_check])
def magic(ctx):
    """Performs all our checking steps: format, lint, type_check."""
    pass

