# main.py
import typer
import os
from pathlib import Path
import shutil
import subprocess
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
import sys # Import sys for cli_command_args logging
import pandas as pd # Needed for data loading display
import re # Needed for parsing agent outputs

# Import Rich components
from rich.traceback import install
install()
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table
from rich.syntax import Syntax # Added for code display
from rich.box import MINIMAL # Added for table display
from typing_extensions import Annotated

# Import agents and their output models from agents.py
from agents import (
    project_initializer_agent, # Agent for project setup (now a BaseModel instance)
    data_loader_agent,         # Agent for data loading (now a BaseModel instance)
    aether_analysis_agent,     # AI agent for data analysis (now a pydantic_ai.Agent instance)
    solution_generation_agent, # AI agent for solution generation (now a pydantic_ai.Agent instance)
    register_file_auto_describe_agent, # AI agent for auto-describing files (now a pydantic_ai.Agent instance)
    aether_insight_agent,      # Placeholder AI agent (now a pydantic_ai.Agent instance)
    aether_code_generation_agent, # Placeholder AI agent (now a pydantic_ai.Agent instance)
    aether_report_agent,       # Placeholder AI agent (now a pydantic_ai.Agent instance)
    aether_explain_agent,      # Placeholder AI agent (now a pydantic_ai.Agent instance)
    aether_hypothesize_agent,  # Placeholder AI agent (now a pydantic_ai.Agent instance)
    aether_command_suggest_agent, # Placeholder AI agent (now a pydantic_ai.Agent instance)
    
    # Import output models for type hinting and display
    ProjectStructureOutput,
    RegisterFileAutoDescribeOutput,
    AetherInsightOutput,
    AnalysisOutput,
    CodeGenerationOutput,
    MarkdownReportOutput,
    SolutionOutput, # Custom output model for this project's solution
    AetherExplainOutput,
    AetherHypothesizeOutput,
    AetherCommandSuggestOutput
)

# Initialize Rich Console
console = Console()

app = typer.Typer(help="DSolve CLI: Your intelligent assistant for data science workflows.")

# Define nested typer app for 'generate' command (kept for template consistency)
generate_app = typer.Typer(help="Generate code or a report.")
app.add_typer(generate_app, name="generate")


# --- Project Configuration ---
# Use Path objects for consistency
# Initialize PROJECT_ROOT to None globally.
# This variable will be set by the main_callback or init command.
PROJECT_ROOT: Optional[Path] = None
DSOLVE_DIR_NAME = ".dsolve" # Changed from .aetherstats to .dsolve for this project
MANIFEST_FILE_NAME = "manifest.json" # Kept for consistency if manifest logic is added later
LOG_FILE_NAME = "logs.txt"

# --- Utility Functions ---

def _find_project_root() -> Optional[Path]:
    """
    Traverses up the directory tree to find the project root.
    A project root is identified by the presence of a '.dsolve' directory.
    """
    current_dir = Path.cwd()
    # Check current_dir and then its parents up to the file system root
    while True:
        if (current_dir / DSOLVE_DIR_NAME).is_dir():
            return current_dir
        if current_dir == current_dir.parent: # Reached file system root
            break
        current_dir = current_dir.parent
    return None # Not in an DSolve project

def _get_manifest_path(project_root: Path) -> Path:
    """Returns the path to the project's manifest file."""
    return project_root / DSOLVE_DIR_NAME / MANIFEST_FILE_NAME

def _load_manifest(project_root: Path) -> List[dict]:
    """Loads the project manifest, creating an empty one if it doesn't exist."""
    manifest_path = _get_manifest_path(project_root)
    if not manifest_path.exists():
        return []
    try:
        with open(manifest_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        console.print(f"[bold red]Warning:[/bold red] Manifest file '{manifest_path}' is corrupted. Starting with empty manifest.")
        return []

def _save_manifest(project_root: Path, manifest_data: List[dict]) -> None:
    """Saves the project manifest."""
    manifest_path = _get_manifest_path(project_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True) # Ensure .dsolve dir exists
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=4)

def _get_log_file_path(project_root: Path) -> Path:
    """Returns the full path to the logs.txt file."""
    return project_root / DSOLVE_DIR_NAME / LOG_FILE_NAME

def _log_interaction(
    project_root: Path,
    action_type: str,
    cli_command_args: List[str], # The arguments directly from sys.argv[1:] or constructed for replay
    agent_output: Optional[Any] = None, # The output model from an agent call
    file_saved_path: Optional[Path] = None, # Path of any file created/modified by the action
    message: Optional[str] = None, # A general message for the log entry
    file_content_snapshot: Optional[Dict[str, str]] = None # Restored: Optional dictionary for file content snapshots
):
    """
    Logs the command, action type, any agent output, and any saved file paths as a JSON object per line.
    This log is designed to be replayable.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "action_type": action_type,
        "cli_command_args": cli_command_args,
        "message": message,
        "agent_output": agent_output.model_dump() if hasattr(agent_output, 'model_dump') else None, # Use model_dump for Pydantic models
        "file_saved_path": str(file_saved_path) if file_saved_path else None,
        "file_content_snapshot": file_content_snapshot # Restored: Include file content snapshot
    }
    log_file_path = _get_log_file_path(project_root)
    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + "\n")
        console.print(f"[dim]Logged action '{action_type}' to {log_file_path}[/dim]")
    except Exception as e:
        console.print(f"[bold red]Error logging interaction: {e}[/bold red]")

def _get_file_summary_for_ai(file_path: Path) -> str:
    """
    Generates a text summary of a file for processing by AI's intelligent features.
    Handles basic text files (CSV, TXT, MD, PY, JSON) by reading first few lines.
    For other types, provides basic info.
    """
    try:
        if file_path.suffix.lower() in ['.csv', '.txt', '.md', '.py', '.json', '.xlsx', '.xls']:
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                # For Excel, try to read a small sample if pandas is available
                try:
                    df_sample = pd.read_excel(file_path, nrows=5)
                    content_sample = df_sample.to_markdown(index=False)
                    summary = f"File Type: {file_path.suffix.upper()} (Excel) File\n"
                    summary += f"File Name: {file_path.name}\n"
                    summary += f"Size: {file_path.stat().st_size / 1024:.2f} KB\n"
                    summary += f"First few rows:\n{content_sample}\n"
                    return summary
                except Exception as e:
                    return f"Could not read Excel sample for {file_path.name}: {e}. Treating as generic binary."

            else: # CSV, TXT, MD, PY, JSON
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = [f.readline() for _ in range(5)] # Read first 5 lines
                    content_sample = "".join(lines).strip()
                
                summary = f"File Type: {file_path.suffix.upper()} Text File\n"
                summary += f"File Name: {file_path.name}\n"
                summary += f"Size: {file_path.stat().st_size / 1024:.2f} KB\n"
                if file_path.suffix.lower() == '.csv':
                    # Try to extract headers for CSV
                    headers = content_sample.split('\n')[0].strip()
                    summary += f"CSV Headers: {headers}\n"
                    summary += f"First few lines:\n{content_sample}\n"
                else:
                    summary += f"Content Sample:\n{content_sample}\n"
                return summary
        else:
            # For binary or unknown files
            return (
                f"File Type: Binary/Unknown ({file_path.suffix.upper()})\n"
                f"File Name: {file_path.name}\n"
                f"Size: {file_path.stat().st_size / 1024:.2f} KB\n"
                "Content cannot be directly read for analysis. Please provide a manual description."
            )
    except Exception as e:
        return f"Could not generate summary for {file_path.name}: {e}"

def _get_file_content_for_log(file_path: Path) -> Optional[str]:
    """
    Reads the content of a text file for logging.
    Returns None if the file is too large or not a readable text file.
    """
    MAX_LOG_FILE_SIZE_KB = 500 # Max 500KB to log file content
    if file_path.stat().st_size > MAX_LOG_FILE_SIZE_KB * 1024:
        console.print(f"[bold yellow]Warning:[/bold yellow] File '{file_path.name}' is too large ({file_path.stat().st_size / 1024:.2f} KB) to log its full content. Skipping.", style="bold yellow")
        return None
    
    try:
        # Attempt to read as UTF-8 text
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except UnicodeDecodeError:
        console.print(f"[bold yellow]Warning:[/bold yellow] File '{file_path.name}' is not a readable text file. Skipping content logging.", style="bold yellow")
        return None
    except Exception as e:
        console.print(f"[bold red]Error reading file '{file_path.name}' for logging: {e}[/bold red]", style="bold red")
        return None


def _run_uv_command(cwd: Path, args: List[str]) -> bool:
    """Helper to run uv commands with Rich status."""
    command = ["uv"] + args
    try:
        with console.status(f"[bold blue]Running uv command: {' '.join(command)}[/bold blue]"):
            result = subprocess.run(
                command,
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "VIRTUAL_ENV_PROMPT": f"({cwd.name})"} # Set prompt for uv venv
            )
            console.print(f"[green]uv output:[/green]\n[dim]{result.stdout}[/dim]")
            if result.stderr:
                console.print(f"[yellow]uv warnings/errors:[/yellow]\n[dim]{result.stderr}[/dim]")
            return True
    except FileNotFoundError:
        console.print("[bold red]Error: 'uv' command not found.[/bold red]")
        console.print("Please install uv: [cyan]pip install uv[/cyan] or [cyan]curl -LsSf https://astral.sh/uv/install.sh | sh[/cyan]")
        return False
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error running uv command: {e}[/bold red]")
        console.print(f"[red]STDOUT:[/red]\n[dim]{e.stdout}[/dim]")
        console.print(f"[red]STDERR:[/red]\n[dim]{e.stderr}[/dim]")
        return False
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred while running uv: {e}[/bold red]")
        return False

def _run_git_command(cwd: Path, args: List[str]) -> subprocess.CompletedProcess:
    """Helper to run git commands in the specified directory."""
    command = ["git"] + args
    
    with console.status(f"[bold blue]Running git command: {' '.join(command)}[/bold blue]"):
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                check=True,  # Raise CalledProcessError for non-zero exit codes
                capture_output=True,
                text=True,    # Decode stdout/stderr as text
                encoding='utf-8' # Explicitly set encoding for consistent output
            )
            return result
        except FileNotFoundError:
            console.print("[bold red]Error: Git command not found. Please ensure Git is installed and in your PATH.[/bold red]")
            raise # Re-raise to indicate critical failure
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]Error running git command: {e}[/bold red]")
            console.print(f"[red]STDOUT:[/red]\n[dim]{e.stdout}[/dim]")
            console.print(f"[red]STDERR:[/red]\n[dim]{e.stderr}[/dim]")
            raise # Re-raise to allow upstream handling


# --- Component Templates and Mappings (Kept for template consistency) ---
_COMPONENT_TYPES = {
    "script": {
        "dir": "scripts",
        "extension": ".py",
        "template": "#!/usr/bin/env python\n# DSolve Generated Script\n\nimport pandas as pd\n\ndef main():\n    # Your analysis code here\n    pass\n\nif __name__ == '__main__':\n    main()\n"
    },
    "notebook": {
        "dir": "notebooks",
        "extension": ".ipynb",
        "template": """{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# DSolve Generated Jupyter Notebook"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": ""
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}
"""
    },
    "report": {
        "dir": "reports",
        "extension": ".md",
        "template": "# DSolve Report: {name_title}\n\n## Overview\n\nThis report summarizes the findings for the '{name}' analysis.\n\n## Data Used\n\n## Methodology\n\n## Results\n\n## Conclusion\n\n---\n*Generated by DSolve on {current_time_str}.*\n"
    },
    "task": {
        "dir": "metadata/tasks",
        "extension": ".md",
        "template": "# Task: {name_title}\n\n- **Status:** [ ] To Do\n- **Created:** {current_time_str}\n- **Assigned To:** \n\n## Description\n\nWrite a detailed description of the task here.\n\n## Steps\n\n- [ ] Step 1\n- [ ] Step 2\n\n"
    }
}

# Define valid data types for registration (Kept for template consistency)
_REGISTER_DATA_TYPES = ["raw_data", "processed_data", "model", "report", "script_output", "other"]


# --- Project Management Commands ---

@app.callback()
def main_callback(ctx: typer.Context):
    """
    Initializes the project root before any command runs.
    """
    global PROJECT_ROOT # Declare global to modify the module-level variable
    # Only 'solve' and 'init' commands handle project initialization themselves
    if ctx.invoked_subcommand not in ["solve", "init", "guide"]:
        PROJECT_ROOT = _find_project_root()
        if PROJECT_ROOT is None:
            console.print("Error: Not within a DSolve project. Please run '[cyan]dsolve init <project_name>[/cyan]' or '[cyan]dsolve solve <file_path> <target_feature>[/cyan]' to start a new project.")
            raise typer.Exit(code=1)

# The 'init' command from your template (kept for consistency, though 'solve' is primary for this project)
@app.command(
    "init",
    help="Initializes a new DSolve project with a standardized directory structure, Git, and uv virtual environment."
)
def init_project(
    project_name: str = typer.Argument(
        ...,
        help="The name of the new DSolve project. A directory with this name will be created."
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Overwrite existing project directory if it exists. WARNING: This will delete existing files!"
    ),
    no_git: bool = typer.Option(
        False,
        "--no-git",
        help="Do not initialize a Git repository in the new project."
    ),
    no_uv: bool = typer.Option(
        False,
        "--no-uv",
        help="Do not initialize a 'uv' virtual environment in the new project."
    )
) -> None:
    """
    Initializes a new DSolve project.

    This command creates a new directory with the given project_name and
    sets up a standardized folder structure for data, scripts, reports, etc.
    It also creates a hidden '.dsolve' directory to mark the project root.
    Optionally initializes a Git repository and a uv virtual environment.
    """
    global PROJECT_ROOT # Add global declaration here as well, since init can set it.
    project_path = Path.cwd() / project_name

    if project_path.exists():
        if force:
            if not Confirm.ask(
                f"[bold yellow]The directory '{project_name}' already exists.[/bold yellow] "
                "Are you sure you want to [bold red]delete its contents[/bold red] and re-initialize?",
                default=False, console=console
            ):
                console.print("[bold red]Aborting initialization.[/bold red]")
                raise typer.Exit(code=1)
            try:
                with console.status(f"[bold blue]Clearing existing directory '{project_name}'...[/bold blue]"):
                    shutil.rmtree(project_path)
                console.print(f"[bold green]Existing directory '{project_name}' cleared.[/bold green]")
            except OSError as e:
                console.print(f"[bold red]Error: Could not remove existing directory '{project_name}': {e}[/bold red]", highlight=False)
                raise typer.Exit(code=1)
        else:
            console.print(
                f"[bold red]Error: Directory '{project_name}' already exists. "
                "Use --force to overwrite.[/bold red]", highlight=False
            )
            raise typer.Exit(code=1)

    try:
        with Live(Spinner("dots", text=Text("Initializing DSolve project...", style="bold cyan")),
                                     transient=True, console=console) as live:
            
            # Create main project directory
            project_path.mkdir(parents=True, exist_ok=False)
            live.console.print(f"[bold green]Created project directory:[/bold green] {project_path}")

            # Define the subdirectories
            subdirectories = [
                "data/raw",
                "data/processed",
                "notebooks",
                "scripts",
                "reports",
                "models",
                "metadata",
                "metadata/tasks",
                "config",
                DSOLVE_DIR_NAME # Hidden directory to mark the project root
            ]

            # Create subdirectories
            for subdir in subdirectories:
                (project_path / subdir).mkdir(parents=True, exist_ok=True)
                live.console.print(f"[green]Created directory:[/green] {project_path / subdir}")

            # Define files to create
            current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")

            readme_content = f"""# {project_name.replace('_', ' ').title()} DSolve Project

An DSolve project for: **{project_name.replace('_', ' ').title()}**.

This project provides a standardized structure for your data science analyses.

## Directory Structure:
- `data/raw/`: Original, immutable datasets.
- `data/processed/`: Cleaned, transformed, or derived datasets.
- `notebooks/`: Jupyter notebooks or interactive exploration files.
- `scripts/`: Production-ready scripts for data processing, analysis, and modeling.
- `reports/`: Generated reports, visualizations, presentations, and final documents.
- `models/`: Saved trained statistical models.
- `metadata/`: Project-specific configuration, task lists, or other metadata files.
- `metadata/tasks/`: Markdown files for individual tasks.
- `config/`: Project-specific configuration files.
- `.{DSOLVE_DIR_NAME}/`: Internal DSolve project markers and configurations (do not modify manually).
- `.venv/`: [bold green]uv[/bold green] virtual environment (if initialized).

## Getting Started:
1. Navigate into the project directory: [bold cyan]cd {project_name}[/cyan]
2. [bold green]If using uv:[/bold green] Activate the environment: `source .venv/bin/activate` (Linux/macOS) or `.venv\\Scripts\\activate` (Windows PowerShell).
3. Install dependencies: `pip install -r requirements.txt` (or `uv pip install -r requirements.txt`)
4. Use DSolve commands to manage your workflow, e.g., `dsolve solve data/raw/my_dataset.csv "target_feature"`.

---
*Generated by DSolve at {current_time_str}.*
"""

            gitignore_content = f"""
# DSolve specific ignores
/{DSOLVE_DIR_NAME}/
/models/
/data/processed/ # Often large and reproducible from raw + scripts
/reports/*.pdf   # Generated PDFs
/reports/*.html  # Generated HTML reports

# uv virtual environment
/.venv/

# Common ignores
*.pyc
__pycache__/
.env
.DS_Store
*.ipynb_checkpoints/
"""
            files_to_create = {
                "README.md": readme_content,
                ".gitignore": gitignore_content,
                "requirements.txt": "pandas\nnumpy\nscipy\nmatplotlib\nseaborn\nscikit-learn\nrich\ntyper\npydantic\npydantic-ai\nopenpyxl\nxlrd\n" # Basic common deps
            }

            # Create files
            for name, content in files_to_create.items():
                file_path = project_path / name
                with open(file_path, "w") as f:
                    f.write(content)
                live.console.print(f"[green]Created file:[/green] {file_path}")
            
            # Create initial manifest file
            _save_manifest(project_path, [])
            live.console.print(f"[green]Created manifest file:[/green] {_get_manifest_path(project_path)}")
            # Create initial log file
            (_get_log_file_path(project_path)).touch()
            live.console.print(f"[green]Created log file:[/green] {_get_log_file_path(project_path)}")


        # Initialize Git repository
        if not no_git:
            try:
                # Change directory to the new project root before initializing Git
                original_cwd = Path.cwd() # Save original cwd
                os.chdir(project_path)
                
                # Use the _run_git_command helper here
                _run_git_command(project_path, ["init"])
                console.print("[bold blue]Initialized empty Git repository.[/bold blue]")

                _run_git_command(project_path, ["add", "."])
                _run_git_command(project_path, ["commit", "-m", "Initial DSolve project setup"])
                console.print("[bold blue]Performed initial Git commit.[/bold blue]")
                
            except Exception as e: # Catch any exception from _run_git_command
                console.print(f"[bold yellow]Warning: Git initialization failed: {e}[/bold yellow]", highlight=False)
                console.print("[bold yellow]Please ensure Git is installed and configured correctly.[/bold yellow]")
            finally:
                # Change back to the original directory
                os.chdir(original_cwd)
        
        # Initialize uv virtual environment
        if not no_uv:
            console.print("[dim]---[/dim]")
            console.print("[bold blue]Setting up uv virtual environment...[/bold blue]")
            
            original_cwd = Path.cwd()
            os.chdir(project_path)
            
            # Check if .venv already exists (shouldn't happen on new init)
            if not (project_path / ".venv").is_dir():
                if _run_uv_command(project_path, ["venv", ".venv"]):
                    console.print("[bold green]uv virtual environment created at ./.venv[/bold green]")
                else:
                    console.print("[bold red]Failed to create uv virtual environment.[/bold red]")
                    os.chdir(original_cwd)
                    raise typer.Exit(code=1)
            else:
                console.print("[dim]uv virtual environment already exists. Skipping creation.[/dim]")

            # Always attempt to install dependencies if requirements.txt exists
            if (project_path / "requirements.txt").exists():
                if _run_uv_command(project_path, ["pip", "install", "-r", "requirements.txt"]):
                    console.print("[bold green]Initial dependencies installed via uv.[/bold green]")
                else:
                    console.print("[bold red]Failed to install initial dependencies with uv.[/bold red]")
            else:
                console.print("[dim]No requirements.txt found. Skipping initial dependency install.[/dim]")

            os.chdir(original_cwd)

        console.print(
            Panel(
                f"[bold green]Successfully initialized DSolve project '{project_name}'![/bold green]\n\n"
                f"Your project is located at: [cyan]{project_path}[/cyan]\n"
                "Navigate into your project directory using: [bold cyan]cd {project_name}",
                title="[bold green]Project Initialized[/bold green]",
                border_style="green"
            )
        )
        if not no_git:
            console.print("[bold blue]A Git repository has been initialized. Don't forget to commit your changes![/bold blue]")
        
        console.print("[bold yellow]Remember to activate your virtual environment for local development:[/bold yellow]")
        console.print(f"  [cyan]cd {project_name}[/cyan]")
        console.print("  [cyan]source .venv/bin/activate[/cyan] (Linux/macOS)")
        console.print("  [cyan].venv\\Scripts\\activate[/cyan] (Windows PowerShell)")

        # Log the init action *after* successful initialization
        # Set PROJECT_ROOT here as well, for commands that might run immediately after init
        PROJECT_ROOT = project_path
        _log_interaction(
            project_path,
            "init",
            cli_command_args=["init", project_name, "--force"] if force else ["init", project_name],
            message=f"Project '{project_name}' initialized."
        )

    except OSError as e:
        console.print(f"[bold red]Error initializing project '{project_name}': {e}[/bold red]", highlight=False)
        # Clean up partially created directory if an error occurred
        if project_path.exists() and not force:
            try:
                shutil.rmtree(project_path)
                console.print(f"[bold red]Partially created directory '{project_name}' removed due to error.[/bold red]", highlight=False)
            except OSError as cleanup_e:
                console.print(f"[bold red]Error during cleanup: {cleanup_e}[/bold red]", highlight=False)
        raise typer.Exit(code=1)


@app.command(
    "solve",
    help="Initializes a new data science project, loads data, and provides AI-driven analysis and solution proposals."
)
def solve(
    file_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
            help="Path to the input data file (e.g., CSV, Excel)."
        )
    ],
    target_feature: Annotated[
        str,
        typer.Argument(
            help="The name of the target feature (column) for prediction or analysis."
        )
    ],
    project_name: Annotated[
        Optional[str],
        typer.Option(
            "--name", "-n",
            help="Optional: Name for the new data science project directory. Defaults to 'dsolve_project_[filename_without_extension]'."
        )
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force", "-f",
            help="Overwrite existing project directory if it exists. WARNING: This will delete existing files!"
        )
    ] = False,
):
    """
    Sets up a data science project for the given file and target,
    then uses AI to analyze the data and propose a solution.
    """
    global PROJECT_ROOT

    # Determine the project name if not provided
    if project_name is None:
        # Replace non-alphanumeric characters in file_path.stem for a clean project name
        project_name = f"dsolve_project_{''.join(c for c in file_path.stem if c.isalnum() or c == '_').strip('_')}"

    project_path = Path.cwd() / project_name

    # Handle existing directory
    if project_path.exists():
        if force:
            if not Confirm.ask(
                f"[bold yellow]The directory '{project_name}' already exists.[/bold yellow]\n"
                "Are you sure you want to [bold red]delete its contents[/bold red] and re-initialize?",
                default=False, console=console
            ):
                console.print("[bold red]Aborting initialization.[/bold red]")
                raise typer.Exit(code=1)
            try:
                with console.status(f"[bold blue]Clearing existing directory '{project_name}'...[/bold blue]"):
                    shutil.rmtree(project_path)
                console.print(f"[bold green]Existing directory '{project_name}' cleared.[/bold green]")
            except OSError as e:
                console.print(f"[bold red]Error: Could not remove existing directory '{project_name}': {e}[/bold red]", highlight=False)
                raise typer.Exit(code=1)
        else:
            console.print(
                f"[bold red]Error: Directory '{project_name}' already exists. "
                "Use --force to overwrite.[/bold red]", highlight=False
            )
            raise typer.Exit(code=1)

    try:
        with Live(Spinner("dots", text=Text("Initializing DSolve project...", style="bold blue")), console=console) as live:
            # Create project root and .dsolve directory
            project_path.mkdir(parents=True, exist_ok=True)
            dsolve_dir = project_path / DSOLVE_DIR_NAME
            dsolve_dir.mkdir(exist_ok=True)
            _get_log_file_path(project_path).touch(exist_ok=True)
            PROJECT_ROOT = project_path # Set global PROJECT_ROOT

            live.text = Text("Instantiating Agents...", style="bold blue")
            # Agents are imported as already-instantiated objects from agents.py

            # 1. Generate the project folder structure
            live.text = Text(f"Generating project structure for '{project_name}'...", style="bold blue")
            # Call run_sync on the ProjectInitializer agent instance
            project_struct_output: ProjectStructureOutput = project_initializer_agent.run_sync(base_path=project_path)
            live.console.print(f"[bold green]Project structure created at: {project_struct_output.project_path}[/bold green]")
            
            # Log project initialization
            _log_interaction(
                project_root=PROJECT_ROOT,
                action_type="project_init",
                cli_command_args=["solve", str(file_path), target_feature, f"--name={project_name}", f"--force={force}"],
                agent_output=project_struct_output,
                message=f"DSolve project '{project_name}' initialized."
            )

            # 2. Copy the input data file into the new project's 'data/raw' directory
            destination_data_path = project_struct_output.data_raw_path / file_path.name
            live.text = Text(f"Copying data file from '{file_path}' to '{destination_data_path}'...", style="bold blue")
            shutil.copy(file_path, destination_data_path)
            live.console.print("Data file copied successfully.")

            # Log data file copy
            _log_interaction(
                project_root=PROJECT_ROOT,
                action_type="data_copy",
                cli_command_args=["solve", str(file_path), target_feature, f"--name={project_name}", f"--force={force}"],
                file_saved_path=destination_data_path,
                message=f"Copied data file {file_path.name} to project."
            )

            # 3. Load the data using the agent
            live.text = Text("Loading data...", style="bold blue")
            # Call run_sync on the DataLoader agent instance
            df = data_loader_agent.run_sync(file_path=destination_data_path)
            live.console.print("Data loaded successfully. Displaying first 5 rows:")
            live.console.print(Panel(df.head().to_markdown(index=False), title="[bold cyan]Data Sample[/bold cyan]", highlight=True))

            # 4. Perform AI-driven data analysis
            live.text = Text("Sending data for AI analysis... (This may take a moment)", style="bold blue")
            data_sample = df.head(5).to_markdown(index=False)
            
            # Capture df.info() into a string
            info_string_stream = pd.io.formats.info.StringIO()
            df.info(buf=info_string_stream)
            data_info_str = info_string_stream.getvalue()

            # Simplified prompt content as the agent's system_prompt handles the output format
            analysis_prompt_content = f"""
            Analyze the following dataset sample and its information,
            and provide a comprehensive analysis report.
            The primary goal is to understand the dataset with respect to the target feature '{target_feature}'.

            Dataset Sample (first 5 rows):
            ```
            {data_sample}
            ```

            Dataset Info (output of df.info()):
            ```
            {data_info_str}
            ```

            Based on this data, provide a detailed analysis including:
            1.  **Initial Observations**: Data types, number of entries, non-null counts, memory usage.
            2.  **Feature Characteristics**: Briefly describe key features, their potential roles, and any immediate patterns.
            3.  **Target Feature Analysis**: What kind of problem is this likely to be (classification, regression, etc.) based on the target? What are its properties?
            4.  **Potential Challenges**: Discuss missing values, data inconsistencies, outliers, categorical encoding needs, scaling requirements, data imbalance, or feature interactions.
            5.  **Exploratory Data Analysis (EDA) Suggestions**: Propose specific visualizations (histograms, scatter plots, correlation matrices), statistical tests, or grouping analyses that would be beneficial.
            6.  **Initial Model Thoughts**: Suggest a few suitable machine learning models for predicting '{target_feature}' and briefly justify why they might be a good fit.
            """
            # Call run_sync on the aether_analysis_agent instance
            analysis_result = aether_analysis_agent.run_sync(analysis_prompt_content)
            analysis_output: AnalysisOutput = analysis_result.output # Directly get the Pydantic model
            
            live.console.print("\n--- [bold green]AI Data Analysis Results[/bold green] ---")
            # Display analysis output based on the new AnalysisOutput structure
            for analysis_res in analysis_output.analyses:
                live.console.print(Panel(
                    f"[bold yellow]Test Name:[/bold yellow] {analysis_res.test_name}\n"
                    f"[bold yellow]Summary:[/bold yellow] {analysis_res.summary}\n"
                    f"[bold yellow]P-Value:[/bold yellow] {analysis_res.p_value if analysis_res.p_value is not None else 'N/A'}\n"
                    f"[bold yellow]Interpretation:[/bold yellow] {analysis_res.interpretation}\n"
                    f"[bold yellow]Key Metrics:[/bold yellow] {json.dumps(analysis_res.key_metrics, indent=2)}\n"
                    f"[bold yellow]Assumptions Notes:[/bold yellow] {analysis_res.assumptions_notes if analysis_res.assumptions_notes else 'None'}",
                    title=f"[bold cyan]Analysis: {analysis_res.test_name}[/bold cyan]", highlight=True
                ))
            live.console.print(f"[bold green]Overall Conclusion:[/bold green] {analysis_output.overall_conclusion}")
            if analysis_output.suggested_follow_up:
                live.console.print("[bold cyan]Suggested Follow-up:[/bold cyan]")
                for step in analysis_output.suggested_follow_up:
                    live.console.print(f"- {step}")
            live.console.print("--------------------------------")

            # Log AI analysis
            _log_interaction(
                project_root=PROJECT_ROOT,
                action_type="ai_analysis",
                cli_command_args=["solve", str(file_path), target_feature, f"--name={project_name}", f"--force={force}"],
                agent_output=analysis_output,
                message="AI data analysis performed."
            )

            # 5. Generate AI-driven solution proposal
            live.text = Text("Generating AI-driven solution proposal... (This may take a moment)", style="bold blue")
            # Simplified prompt content as the agent's system_prompt handles the output format
            solution_prompt_content = f"""
            Based on the following comprehensive data analysis results, propose a detailed, step-by-step plan to solve the data science problem, which is to predict or analyze '{target_feature}'.
            The plan should be actionable and cover the typical stages of a data science project, considering best practices.

            Data Analysis Results:
            ```
            {json.dumps(analysis_output.model_dump(), indent=2)}
            ```

            Your proposed plan should include the following sections in detail:
            1.  **Project Setup & Environment**: Recommendations for project structure (if not already set up), dependency management, and version control.
            2.  **Data Ingestion & Validation**: How to load the data robustly, including initial validation checks (e.g., schema, data types, ranges).
            3.  **Data Preprocessing**: Specific, actionable steps for handling:
                * Missing Values (e.g., imputation strategies: mean, median, mode, predictive)
                * Outliers (detection and handling methods)
                * Categorical Features (encoding: One-Hot, Label, Target, etc.)
                * Numerical Features (scaling: Standard, Min-Max, Robust; transformations: log, power)
                * Date/Time features (extraction of components)
            4.  **Feature Engineering**: Concrete ideas for creating new, more informative features from existing ones (e.g., interaction terms, polynomial features, aggregation features).
            5.  **Model Selection & Justification**: A more refined justification for choosing specific machine learning models (e.g., Linear Regression, Logistic Regression, Decision Trees, Random Forests, Gradient Boosting, SVM, Neural Networks) based on the problem type and data characteristics identified in the analysis. Consider ensemble methods.
            6.  **Model Training & Evaluation Strategy**: How to split data (e.g., train-test split, cross-validation), specific metrics to use for evaluation (e.g., R2, MAE, RMSE for regression; Accuracy, Precision, Recall, F1, ROC-AUC for classification), and hyperparameter tuning approaches.
            7.  **Model Interpretation & Explainability**: How to understand why the model makes certain predictions (e.g., SHAP, LIME, feature importances).
            8.  **Deployment Considerations (Conceptual)**: Brief thoughts on how the trained model might be deployed, potential infrastructure, and monitoring.
            9.  **Further Steps & Iteration**: Any additional recommendations for future work, model improvement, or continuous integration/delivery (CI/CD) aspects.
            """
            # Call run_sync on the SolutionGenerationAgent instance
            solution_result = solution_generation_agent.run_sync(solution_prompt_content)
            solution_output: SolutionOutput = solution_result.output
            
            live.console.print("\n--- [bold green]AI Solution Proposal[/bold green] ---")
            live.console.print(Panel(solution_output.solution_plan, title="[bold yellow]Solution Plan[/bold yellow]", highlight=True))
            if solution_output.steps:
                live.console.print("[bold cyan]Key Steps:[/bold cyan]")
                for step in solution_output.steps:
                    live.console.print(f"- {step}")
            if solution_output.notes:
                live.console.print("[bold cyan]Notes:[/bold cyan]")
                for note in solution_output.notes:
                    live.console.print(f"- {note}")
            live.console.print("----------------------------")

            # Log AI solution
            _log_interaction(
                project_root=PROJECT_ROOT,
                action_type="ai_solution",
                cli_command_args=["solve", str(file_path), target_feature, f"--name={project_name}", f"--force={force}"],
                agent_output=solution_output,
                message="AI solution proposed."
            )

        console.print(Panel(
            f"[bold green]Project '{project_name}' initialized successfully with AI insights![/bold green]\n"
            f"Location: [cyan]{project_path.resolve()}[/cyan]\n\n"
            f"Your data has been copied to: [cyan]{destination_data_path}[/cyan]\n"
            "Review the AI analysis and solution above to continue your work."
        ))

    except ValueError as ve:
        console.print(f"[bold red]Error: {ve}[/bold red]", err=True)
        typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        if project_path.exists() and project_path.is_dir():
            if Confirm.ask(f"[bold yellow]An error occurred during process. Do you want to remove the partially created project directory '{project_name}'?[/bold yellow]", default=True, console=console):
                shutil.rmtree(project_path)
                console.print(f"[bold green]Partially created project '{project_name}' removed.[/bold green]")
        raise typer.Exit(code=1)


@app.command(
    "create",
    help="Creates a new component (script, notebook, report, task) within the current DSolve project."
)
def create_component(
    component_type: str = typer.Argument(
        ...,
        help=f"The type of component to create. Choose from: {', '.join(_COMPONENT_TYPES.keys())}"
    ),
    name: str = typer.Argument(
        ...,
        help="The name of the component (e.g., 'data_cleaning' or 'quarterly_report')."
    )
) -> None:
    """
    Creates a new project component like a script, notebook, report, or task.
    """
    # project_root is guaranteed to be set by main_callback
    project_root = PROJECT_ROOT

    # 2. Validate component type
    component_info = _COMPONENT_TYPES.get(component_type.lower())
    if not component_info:
        console.print(
            f"[bold red]Error:[/bold red] Invalid component type '[bold yellow]{component_type}[/bold yellow]'.\n"
            f"Valid types are: [bold cyan]{', '.join(_COMPONENT_TYPES.keys())}[/cyan]."
        )
        raise typer.Exit(code=1)

    target_dir = project_root / component_info["dir"]
    file_name = f"{name}{component_info['extension']}"
    file_path = target_dir / file_name

    # 3. Check if component already exists
    if file_path.exists():
        console.print(
            f"[bold yellow]Warning:[/bold yellow] A {component_type} named '[bold cyan]{name}[/bold cyan]' already exists at:\n"
            f"{file_path}\n"
            "Consider choosing a different name or manually deleting the existing file if you wish to replace it."
        )
        raise typer.Exit(code=1)

    # 4. Create the component
    try:
        with console.status(f"[bold blue]Creating {component_type} '{name}'...[/bold blue]"):
            # Ensure the target directory exists (it should from init, but good to be safe)
            target_dir.mkdir(parents=True, exist_ok=True)

            template_content = component_info["template"]
            current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
            name_title = name.replace('_', ' ').title()

            # Format template with dynamic values
            formatted_content = template_content.format(
                name=name,
                name_title=name_title,
                current_time_str=current_time_str
            )

            with open(file_path, "w") as f:
                f.write(formatted_content)
            
        console.print(
            f"[bold green]Successfully created {component_type} '[bold cyan]{name}[/bold cyan]' at:[/bold green] {file_path}"
        )
        console.print(f"You can now edit this file in your favorite editor.")

        # Log the create action
        _log_interaction(
            project_root,
            "create",
            cli_command_args=["create", component_type, name],
            file_saved_path=file_path.relative_to(project_root),
            message=f"Created {component_type} '{name}'."
        )

    except OSError as e:
        console.print(f"[bold red]Error creating {component_type} '{name}': {e}[/bold red]", highlight=False)
        raise typer.Exit(code=1)


@app.command(
    "register",
    help="Registers a file (data, model, report, etc.) within the current DSolve project's manifest."
)
def register_file(
    file_path: str = typer.Argument(
        ...,
        help="The path to the file to register, relative to the project root."
    ),
    data_type: str = typer.Option(
        ...,
        "--type", "-t",
        help=f"The type of data being registered. Choose from: {', '.join(_REGISTER_DATA_TYPES)}"
    ),
    description: Optional[str] = typer.Option(
        None,
        "--description", "-d",
        help="A brief description of the file's content or purpose."
    ),
    auto_describe: bool = typer.Option(
        False,
        "--auto-describe",
        help="Use DSolve's intelligent features to automatically generate a description for the file."
    )
) -> None:
    """
    Registers a file (data, model, report, etc.) within the current DSolve project.
    Metadata about the file is stored in a manifest.json file.
    """
    # project_root is guaranteed to be set by main_callback
    project_root = PROJECT_ROOT

    # 2. Validate file path
    absolute_file_path = Path.cwd() / file_path
    if not absolute_file_path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: '[bold yellow]{file_path}[/bold yellow]'. Please provide a valid path.", highlight=False)
        raise typer.Exit(code=1)
    
    # 3. Validate data type
    if data_type.lower() not in _REGISTER_DATA_TYPES:
        console.print(
            f"[bold red]Error:[/bold red] Invalid data type '[bold yellow]{data_type}[/bold yellow]'.\n"
            f"Valid types are: [bold cyan]{', '.join(_REGISTER_DATA_TYPES)}[/cyan]."
        )
        raise typer.Exit(code=1)

    # 4. Handle auto-description with intelligent features
    auto_desc_output: Optional[RegisterFileAutoDescribeOutput] = None
    if auto_describe:
        if description:
            console.print("[bold yellow]Warning:[/bold yellow] Both --description and --auto-describe were provided. Using auto-generated description.", highlight=False)
            
        file_summary_for_ai = _get_file_summary_for_ai(absolute_file_path)
        
        with console.status("[bold magenta]DSolve is generating file description...[/bold magenta]"):
            try:
                # Call run_sync on the agent instance
                agent_response = register_file_auto_describe_agent.run_sync(file_summary_for_ai)
                auto_desc_output = agent_response.output # Direct Pydantic model output
                
                # Update description, data_type, and tags from auto-generated output
                description = auto_desc_output.description
                data_type = auto_desc_output.data_type
                # Ensure data_type is one of the allowed types, fallback if not
                if data_type.lower() not in _REGISTER_DATA_TYPES:
                    console.print(f"[bold yellow]Warning:[/bold yellow] Auto-generated data type '{data_type}' is not recognized. Falling back to provided type or 'other'.", highlight=False)
                    data_type = data_type if data_type.lower() in _REGISTER_DATA_TYPES else "other"
                
                console.print("[bold green]Auto-description generated successfully![/bold green]")
            except Exception as e:
                console.print(f"[bold red]Error generating auto-description: {e}[/bold red]")
                console.print("[bold yellow]Proceeding with manual description or empty description if none provided.[/bold yellow]")
                description = description # Revert to user-provided or None
    
    # Load existing manifest
    manifest_data = _load_manifest(project_root)

    # Check if the file is already registered and update or add
    file_registered = False
    for entry in manifest_data:
        if entry["path"] == str(absolute_file_path.relative_to(project_root)):
            if not Confirm.ask(
                f"[bold yellow]File '{file_path}' is already registered.[/bold yellow] Do you want to update its entry?",
                default=True, console=console
            ):
                console.print("[bold red]Aborting registration.[/bold red]")
                raise typer.Exit(code=1)
            
            entry["timestamp"] = datetime.now().isoformat()
            entry["data_type"] = data_type
            entry["description"] = description
            entry["tags"] = auto_desc_output.tags if auto_desc_output else [] # Use auto-generated tags if available
            entry["overview"] = auto_desc_output.overview if auto_desc_output else ""
            entry["key_variables"] = auto_desc_output.key_variables if auto_desc_output else []
            entry["observations"] = auto_desc_output.observations if auto_desc_output else []
            entry["potential_issues"] = auto_desc_output.potential_issues if auto_desc_output else []
            entry["suggested_next_steps"] = auto_desc_output.suggested_next_steps if auto_desc_output else []

            file_registered = True
            console.print(f"[bold green]Updated registration for:[/bold green] {file_path}")
            break
    
    if not file_registered:
        new_entry = {
            "path": str(absolute_file_path.relative_to(project_root)), # Store relative path
            "name": absolute_file_path.name,
            "timestamp": datetime.now().isoformat(),
            "data_type": data_type,
            "description": description,
            "tags": auto_desc_output.tags if auto_desc_output else [],
            "overview": auto_desc_output.overview if auto_desc_output else "",
            "key_variables": auto_desc_output.key_variables if auto_desc_output else [],
            "observations": auto_desc_output.observations if auto_desc_output else [],
            "potential_issues": auto_desc_output.potential_issues if auto_desc_output else [],
            "suggested_next_steps": auto_desc_output.suggested_next_steps if auto_desc_output else []
        }
        manifest_data.append(new_entry)
        console.print(f"[bold green]Registered new file:[/bold green] {file_path}")

    _save_manifest(project_root, manifest_data)
    
    # Log the register action
    _log_interaction(
        project_root,
        "register",
        cli_command_args=["register", file_path, "--type", data_type, "--description", description or "None", "--auto-describe"] if auto_describe else ["register", file_path, "--type", data_type, "--description", description or "None"],
        message=f"Registered file '{file_path}'.",
        agent_output=auto_desc_output
    )

@app.command(
    "list",
    help="Lists all registered files in the current DSolve project's manifest."
)
def list_files() -> None:
    """
    Lists all files currently registered in the project's manifest.
    """
    project_root = PROJECT_ROOT
    manifest_data = _load_manifest(project_root)

    if not manifest_data:
        console.print("[bold yellow]No files registered in the manifest yet.[/bold yellow]")
        return

    table = Table(
        title="[bold blue]Registered Files[/bold blue]",
        show_header=True,
        header_style="bold magenta",
        box=MINIMAL
    )
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Path", style="dim", no_wrap=False)
    table.add_column("Type", style="green", no_wrap=True)
    table.add_column("Description", style="white", no_wrap=False)
    table.add_column("Registered On", style="yellow", no_wrap=True)

    for entry in manifest_data:
        # Shorten description for table view if too long
        display_description = entry.get("description", "No description").split('\n')[0]
        if len(display_description) > 70:
            display_description = display_description[:67] + "..."
        
        table.add_row(
            entry.get("name", "N/A"),
            entry.get("path", "N/A"),
            entry.get("data_type", "N/A"),
            display_description,
            datetime.fromisoformat(entry.get("timestamp")).strftime("%Y-%m-%d %H:%M") if entry.get("timestamp") else "N/A"
        )
    
    console.print(table)

    _log_interaction(
        project_root,
        "list_files",
        cli_command_args=["list"],
        message="Listed registered files."
    )

@app.command(
    "describe",
    help="Gets a detailed, AI-generated description of a registered file."
)
def describe_file(
    file_path: Annotated[
        str,
        typer.Argument(
            help="The path to the registered file, relative to the project root."
        )
    ]
) -> None:
    """
    Retrieves and displays a detailed, AI-generated description of a registered file.
    """
    project_root = PROJECT_ROOT
    absolute_file_path = project_root / file_path
    
    if not absolute_file_path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found at '{file_path}'. Make sure it's correct and within the project.", highlight=False)
        raise typer.Exit(code=1)

    manifest_data = _load_manifest(project_root)
    file_entry = next((entry for entry in manifest_data if entry["path"] == str(absolute_file_path.relative_to(project_root))), None)

    if not file_entry:
        console.print(f"[bold red]Error:[/bold red] File '{file_path}' is not registered. Please run '[cyan]dsolve register {file_path} --type <type> --auto-describe[/cyan]' first.", highlight=False)
        raise typer.Exit(code=1)
    
    console.print(Panel(
        f"[bold cyan]Description for:[/bold cyan] {file_entry.get('name', 'N/A')}\n"
        f"[bold cyan]Path:[/bold cyan] {file_entry.get('path', 'N/A')}\n"
        f"[bold cyan]Type:[/bold cyan] {file_entry.get('data_type', 'N/A')}\n"
        f"[bold cyan]Registered On:[/bold cyan] {datetime.fromisoformat(file_entry.get('timestamp')).strftime('%Y-%m-%d %H:%M') if file_entry.get('timestamp') else 'N/A'}\n\n"
        f"[bold yellow]Overview:[/bold yellow] {file_entry.get('overview', 'N/A')}\n\n"
        f"[bold yellow]Key Variables:[/bold yellow]\n" + 
        "\n".join([f"  - {kv['name']}: {kv['description']}" for kv in file_entry.get('key_variables', [])]) + "\n\n" +
        f"[bold yellow]Observations:[/bold yellow]\n" + 
        "\n".join([f"  - {obs}" for obs in file_entry.get('observations', [])]) + "\n\n" +
        f"[bold yellow]Potential Issues:[/bold yellow]\n" +
        "\n".join([f"  - {issue}" for issue in file_entry.get('potential_issues', [])]) + "\n\n" +
        f"[bold yellow]Suggested Next Steps:[/bold yellow]\n" +
        "\n".join([f"  - {step}" for step in file_entry.get('suggested_next_steps', [])]) + "\n\n" +
        f"[bold yellow]Tags:[/bold yellow] {', '.join(file_entry.get('tags', []))}",
        title=f"[bold green]Detailed File Description[/bold green]",
        border_style="green"
    ))

    _log_interaction(
        project_root,
        "describe_file",
        cli_command_args=["describe", file_path],
        message=f"Displayed description for '{file_path}'."
    )

@app.command(
    "explore",
    help="Uses AI to find insights and anomalies in a registered data file."
)
def explore_data(
    file_path: Annotated[
        str,
        typer.Argument(
            help="The path to the registered data file, relative to the project root."
        )
    ]
) -> None:
    """
    Uses the Aether Insight Agent to find and display key insights and anomalies
    from a registered data file.
    """
    project_root = PROJECT_ROOT
    absolute_file_path = project_root / file_path

    if not absolute_file_path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found at '{file_path}'. Make sure it's correct and within the project.", highlight=False)
        raise typer.Exit(code=1)
    
    # Load data using DataLoader agent to get a pandas DataFrame
    try:
        df = data_loader_agent.run_sync(file_path=absolute_file_path)
        data_summary_for_ai = df.describe(include='all').to_markdown() + "\n\n" + df.info(buf=pd.io.formats.info.StringIO()).getvalue()
    except Exception as e:
        console.print(f"[bold red]Error loading data from '{file_path}': {e}[/bold red]", highlight=False)
        raise typer.Exit(code=1)

    with console.status("[bold magenta]DSolve is exploring data for insights and anomalies...[/bold magenta]"):
        try:
            # Call run_sync on the AetherInsightAgent
            insight_result = aether_insight_agent.run_sync(data_summary=data_summary_for_ai)
            insight_output: AetherInsightOutput = insight_result.output

            console.print("\n--- [bold green]AI Data Exploration Results[/bold green] ---")
            console.print(Panel(
                f"[bold yellow]Insights:[/bold yellow]\n" + "\n".join([f"- {i}" for i in insight_output.insights]) + "\n\n" +
                f"[bold yellow]Anomalies:[/bold yellow]\n" + "\n".join([f"- {a}" for a in insight_output.anomalies]) + "\n\n" +
                f"[bold yellow]Key Relationships:[/bold yellow]\n" + "\n".join([f"- {rel['variables']}: {rel['description']}" for rel in insight_output.key_relationships]) + "\n\n" +
                f"[bold yellow]Actionable Recommendations:[/bold yellow]\n" + "\n".join([f"- {rec}" for rec in insight_output.actionable_recommendations]),
                title="[bold cyan]Data Insights & Anomalies[/bold cyan]", highlight=True
            ))
            console.print("--------------------------------")

            _log_interaction(
                project_root,
                "explore_data",
                cli_command_args=["explore", file_path],
                agent_output=insight_output,
                message=f"Explored data for '{file_path}'."
            )
        except Exception as e:
            console.print(f"[bold red]Error generating insights: {e}[/bold red]", highlight=False)
            raise typer.Exit(code=1)

@app.command(
    "analyze",
    help="Uses AI to perform a conceptual statistical analysis on a registered data file."
)
def analyze_data(
    file_path: Annotated[
        str,
        typer.Argument(
            help="The path to the registered data file, relative to the project root."
        )
    ],
    analysis_request: Annotated[
        str,
        typer.Argument(
            help="A natural language description of the statistical analysis to perform (e.g., 'compare sales between regions', 'predict customer churn')."
        )
    ]
) -> None:
    """
    Uses the Aether Analysis Agent to perform a conceptual statistical analysis
    and provide structured results.
    """
    project_root = PROJECT_ROOT
    absolute_file_path = project_root / file_path

    if not absolute_file_path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found at '{file_path}'. Make sure it's correct and within the project.", highlight=False)
        raise typer.Exit(code=1)
    
    # Load data summary for the AI agent
    try:
        df = data_loader_agent.run_sync(file_path=absolute_file_path)
        data_summary_for_ai = df.describe(include='all').to_markdown() + "\n\n" + df.info(buf=pd.io.formats.info.StringIO()).getvalue()
    except Exception as e:
        console.print(f"[bold red]Error loading data from '{file_path}': {e}[/bold red]", highlight=False)
        raise typer.Exit(code=1)

    analysis_prompt_content = f"""
    Data Summary:
    ```
    {data_summary_for_ai}
    ```
    Analysis Request: "{analysis_request}"
    """

    with console.status("[bold magenta]DSolve is performing conceptual analysis...[/bold magenta]"):
        try:
            analysis_result = aether_analysis_agent.run_sync(analysis_prompt_content)
            analysis_output: AnalysisOutput = analysis_result.output

            console.print("\n--- [bold green]AI Conceptual Analysis Results[/bold green] ---")
            for analysis_res in analysis_output.analyses:
                console.print(Panel(
                    f"[bold yellow]Test Name:[/bold yellow] {analysis_res.test_name}\n"
                    f"[bold yellow]Summary:[/bold yellow] {analysis_res.summary}\n"
                    f"[bold yellow]P-Value:[/bold yellow] {analysis_res.p_value if analysis_res.p_value is not None else 'N/A'}\n"
                    f"[bold yellow]Interpretation:[/bold yellow] {analysis_res.interpretation}\n"
                    f"[bold yellow]Key Metrics:[/bold yellow] {json.dumps(analysis_res.key_metrics, indent=2)}\n"
                    f"[bold yellow]Assumptions Notes:[/bold yellow] {analysis_res.assumptions_notes if analysis_res.assumptions_notes else 'None'}",
                    title=f"[bold cyan]Analysis: {analysis_res.test_name}[/bold cyan]", highlight=True
                ))
            console.print(f"[bold green]Overall Conclusion:[/bold green] {analysis_output.overall_conclusion}")
            if analysis_output.suggested_follow_up:
                console.print("[bold cyan]Suggested Follow-up:[/bold cyan]")
                for step in analysis_output.suggested_follow_up:
                    console.print(f"- {step}")
            console.print("--------------------------------")

            _log_interaction(
                project_root,
                "analyze_data",
                cli_command_args=["analyze", file_path, analysis_request],
                agent_output=analysis_output,
                message=f"Performed conceptual analysis on '{file_path}' for request: '{analysis_request}'."
            )
        except Exception as e:
            console.print(f"[bold red]Error performing conceptual analysis: {e}[/bold red]", highlight=False)
            raise typer.Exit(code=1)


@app.command(
    "hypothesize",
    help="Uses AI to generate statistical hypotheses based on a registered data file's characteristics."
)
def hypothesize_data(
    file_path: Annotated[
        str,
        typer.Argument(
            help="The path to the registered data file, relative to the project root."
        )
    ]
) -> None:
    """
    Uses the Aether Hypothesize Agent to generate testable statistical hypotheses
    from a registered data file's characteristics.
    """
    project_root = PROJECT_ROOT
    absolute_file_path = project_root / file_path

    if not absolute_file_path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found at '{file_path}'. Make sure it's correct and within the project.", highlight=False)
        raise typer.Exit(code=1)
    
    # Load data summary for the AI agent
    try:
        df = data_loader_agent.run_sync(file_path=absolute_file_path)
        dataset_characteristics_for_ai = df.describe(include='all').to_markdown() + "\n\n" + df.info(buf=pd.io.formats.info.StringIO()).getvalue()
    except Exception as e:
        console.print(f"[bold red]Error loading data from '{file_path}': {e}[/bold red]", highlight=False)
        raise typer.Exit(code=1)

    with console.status("[bold magenta]DSolve is generating hypotheses...[/bold magenta]"):
        try:
            hypothesize_result = aether_hypothesize_agent.run_sync(dataset_characteristics=dataset_characteristics_for_ai)
            hypothesize_output: AetherHypothesizeOutput = hypothesize_result.output

            console.print("\n--- [bold green]AI Generated Hypotheses[/bold green] ---")
            for i, hypothesis_entry in enumerate(hypothesize_output.hypotheses):
                console.print(Panel(
                    f"[bold yellow]Hypothesis {i+1}:[/bold yellow] {hypothesis_entry.get('hypothesis', 'N/A')}\n"
                    f"[bold yellow]Suggested Test:[/bold yellow] {hypothesis_entry.get('suggested_test', 'N/A')}\n"
                    f"[bold yellow]Reasoning:[/bold yellow] {hypothesis_entry.get('reasoning', 'N/A')}",
                    title=f"[bold cyan]Hypothesis {i+1}[/bold cyan]", highlight=True
                ))
            if hypothesize_output.potential_challenges:
                console.print("[bold cyan]Potential Challenges:[/bold cyan]")
                for challenge in hypothesize_output.potential_challenges:
                    console.print(f"- {challenge}")
            console.print("--------------------------------")

            _log_interaction(
                project_root,
                "hypothesize_data",
                cli_command_args=["hypothesize", file_path],
                agent_output=hypothesize_output,
                message=f"Generated hypotheses for '{file_path}'."
            )
        except Exception as e:
            console.print(f"[bold red]Error generating hypotheses: {e}[/bold red]", highlight=False)
            raise typer.Exit(code=1)


@app.command(
    "suggest-command",
    help="Uses AI to suggest DSolve CLI commands based on your natural language query."
)
def suggest_command(
    user_query: Annotated[
        str,
        typer.Argument(
            help="Your natural language query describing the task (e.g., 'how do I clean missing values?', 'visualize sales by month')."
        )
    ],
    file_path: Annotated[
        Optional[str],
        typer.Option(
            "--file", "-f",
            help="Optional: Path to a registered file to provide context for command suggestions."
        )
    ] = None
) -> None:
    """
    Uses the Aether Command Suggest Agent to translate natural language queries
    into executable DSolve CLI commands.
    """
    project_root = PROJECT_ROOT
    available_data_context = "No specific file context provided."

    if file_path:
        absolute_file_path = project_root / file_path
        if not absolute_file_path.exists():
            console.print(f"[bold yellow]Warning:[/bold yellow] Context file '{file_path}' not found. Suggestions will be general.", highlight=False)
        else:
            try:
                df = data_loader_agent.run_sync(file_path=absolute_file_path)
                available_data_context = f"File '{file_path}' has columns: {', '.join(df.columns.tolist())}. Data info:\n{df.info(buf=pd.io.formats.info.StringIO()).getvalue()}"
            except Exception as e:
                console.print(f"[bold yellow]Warning:[/bold yellow] Could not load context from '{file_path}': {e}. Suggestions will be general.", highlight=False)
                available_data_context = f"File '{file_path}' exists but could not be loaded for context. Error: {e}"

    with console.status("[bold magenta]DSolve is suggesting commands...[/bold magenta]"):
        try:
            suggest_result = aether_command_suggest_agent.run_sync(
                user_query=user_query,
                available_data_context=available_data_context
            )
            suggest_output: AetherCommandSuggestOutput = suggest_result.output

            console.print("\n--- [bold green]AI Suggested Commands[/bold green] ---")
            if suggest_output.suggested_commands:
                console.print("[bold yellow]Suggested Commands:[/bold yellow]")
                for cmd in suggest_output.suggested_commands:
                    console.print(f"  [cyan]{cmd}[/cyan]")
                console.print(f"\n[bold yellow]Explanation:[/bold yellow] {suggest_output.explanation}")
            else:
                console.print("[bold yellow]No specific commands suggested.[/bold yellow]")
            
            if suggest_output.clarification_needed:
                console.print(f"\n[bold red]Clarification Needed:[/bold red] {suggest_output.clarification_needed}")
            console.print("--------------------------------")

            _log_interaction(
                project_root,
                "suggest_command",
                cli_command_args=["suggest-command", user_query, "--file", file_path] if file_path else ["suggest-command", user_query],
                agent_output=suggest_output,
                message=f"Suggested commands for query: '{user_query}'."
            )
        except Exception as e:
            console.print(f"[bold red]Error suggesting commands: {e}[/bold red]", highlight=False)
            raise typer.Exit(code=1)


@generate_app.command(
    "code",
    help="Generates Python code for a specified data science task."
)
def generate_code(
    task_description: Annotated[
        str,
        typer.Argument(
            help="A detailed description of the Python code to generate (e.g., 'a script to clean missing values in column X of my_data.csv')."
        )
    ],
    output_filename: Annotated[
        Optional[str],
        typer.Option(
            "--output", "-o",
            help="Optional: The filename to save the generated code to. Defaults to agent's suggestion."
        )
    ] = None,
    file_path: Annotated[
        Optional[str],
        typer.Option(
            "--file", "-f",
            help="Optional: Path to a registered file to provide context for code generation."
        )
    ] = None
) -> None:
    """
    Generates Python code based on a task description and saves it to a file.
    """
    project_root = PROJECT_ROOT
    code_request_prompt = task_description
    
    if file_path:
        absolute_file_path = project_root / file_path
        if not absolute_file_path.exists():
            console.print(f"[bold yellow]Warning:[/bold yellow] Context file '{file_path}' not found. Code generation might be less specific.", highlight=False)
        else:
            try:
                df = data_loader_agent.run_sync(file_path=absolute_file_path)
                code_request_prompt += f"\n\nContext file '{file_path}' has columns: {', '.join(df.columns.tolist())}. Data info:\n{df.info(buf=pd.io.formats.info.StringIO()).getvalue()}"
            except Exception as e:
                console.print(f"[bold yellow]Warning:[/bold yellow] Could not load context from '{file_path}': {e}. Code generation might be less specific.", highlight=False)
                code_request_prompt += f"\n\nContext file '{file_path}' exists but could not be loaded for context. Error: {e}"

    with console.status("[bold magenta]DSolve is generating code...[/bold magenta]"):
        try:
            code_result = aether_code_generation_agent.run_sync(code_request_prompt)
            code_output: CodeGenerationOutput = code_result.output

            suggested_filename = code_output.filename_suggestion or "generated_script.py"
            filename_to_save = output_filename if output_filename else suggested_filename
            save_path = project_root / "scripts" / filename_to_save

            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                f.write(code_output.code)
            
            console.print(Panel(
                f"[bold green]Code Generated Successfully![/bold green]\n"
                f"Saved to: [cyan]{save_path.relative_to(project_root)}[/cyan]\n\n"
                f"[bold yellow]Explanation:[/bold yellow]\n{code_output.explanation}\n\n"
                f"[bold yellow]Required Packages:[/bold yellow] {', '.join(code_output.required_packages)}",
                title="[bold green]Generated Code Summary[/bold green]",
                border_style="green"
            ))
            console.print(Syntax(code_output.code, "python", theme="monokai", line_numbers=True))

            _log_interaction(
                project_root,
                "generate_code",
                cli_command_args=["generate", "code", task_description, "--output", filename_to_save, "--file", file_path] if file_path else ["generate", "code", task_description, "--output", filename_to_save],
                agent_output=code_output,
                file_saved_path=save_path.relative_to(project_root),
                file_content_snapshot={"code": code_output.code},
                message=f"Generated code for task: '{task_description}'."
            )
        except Exception as e:
            console.print(f"[bold red]Error generating code: {e}[/bold red]", highlight=False)
            raise typer.Exit(code=1)


@generate_app.command(
    "report",
    help="Generates a Markdown report on a specified topic."
)
def generate_report(
    report_topic: Annotated[
        str,
        typer.Argument(
            help="The topic of the report (e.g., 'Customer Churn Analysis', 'Sales Performance Q1')."
        )
    ],
    output_filename: Annotated[
        Optional[str],
        typer.Option(
            "--output", "-o",
            help="Optional: The filename to save the generated report to. Defaults to 'report_<topic>.md'."
        )
    ],
    data_context: Annotated[
        Optional[str],
        typer.Option(
            "--context", "-c",
            help="Optional: Additional data context or analysis results to include in the report."
        )
    ] = None
) -> None:
    """
    Generates a Markdown report based on a topic and optional data context, and saves it to a file.
    """
    project_root = PROJECT_ROOT
    report_prompt_elements = f"Topic: {report_topic}"
    if data_context:
        report_prompt_elements += f"\nData Context: {data_context}"

    with console.status("[bold magenta]DSolve is generating report...[/bold magenta]"):
        try:
            report_result = aether_report_agent.run_sync(report_prompt_elements)
            report_output: MarkdownReportOutput = report_result.output

            suggested_filename = f"report_{report_output.title.replace(' ', '_').lower()}.md" if report_output.title else "generated_report.md"
            filename_to_save = output_filename if output_filename else suggested_filename
            save_path = project_root / "reports" / filename_to_save

            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                f.write(report_output.markdown_content)
            
            console.print(Panel(
                f"[bold green]Report Generated Successfully![/bold green]\n"
                f"Saved to: [cyan]{save_path.relative_to(project_root)}[/cyan]\n\n"
                f"[bold yellow]Title:[/bold yellow] {report_output.title}\n"
                f"[bold yellow]Summary:[/bold yellow]\n{report_output.summary}",
                title="[bold green]Generated Report Summary[/bold green]",
                border_style="green"
            ))
            console.print(Panel(report_output.markdown_content, title="[bold cyan]Report Content[/bold cyan]", highlight=True))

            _log_interaction(
                project_root,
                "generate_report",
                cli_command_args=["generate", "report", report_topic, "--output", filename_to_save, "--context", data_context] if data_context else ["generate", "report", report_topic, "--output", filename_to_save],
                agent_output=report_output,
                file_saved_path=save_path.relative_to(project_root),
                file_content_snapshot={"markdown_content": report_output.markdown_content},
                message=f"Generated report for topic: '{report_topic}'."
            )
        except Exception as e:
            console.print(f"[bold red]Error generating report: {e}[/bold red]", highlight=False)
            raise typer.Exit(code=1)

# Add this new command to your main.py file

@app.command(
    "guide",
    help="Displays helpful guides and documentation for DSolve commands."
)
def show_guide(
    guide_name: Annotated[
        str,
        typer.Argument(
            help="The name of the guide to display. Currently, only 'create-project' is available."
        )
    ]
) -> None:
    """
    Displays helpful guides and documentation for DSolve commands.
    """
    if guide_name.lower() == "create-project":
        guide_content = r"""Creating a DSolve Project: A Step-by-Step Guide
DSolve is a powerful command-line interface designed to streamline your data science workflows by automating project setup, data analysis, and solution planning with the help of AI.

Prerequisites
Before you begin, ensure you have:

Python 3.8+ installed on your system.

uv (recommended for virtual environment and package management) or pip installed.

1. Installing DSolve
First, you need to install the DSolve CLI. You can go to its website and follow the tutorial

Once installed, you should be able to run dsolve --help to see the available commands.

2. Creating a New Project
DSolve offers two primary ways to start a new project:

Option A: Create a Project with Initial Data and AI Analysis (dsolve solve) - Recommended
This is the most comprehensive way to start, as it not only sets up your project but also immediately processes your data with AI to provide initial analysis and a solution plan.

Command Structure:

dsolve solve <file_path> <target_feature> [--name <project_name>] [--force]

<file_path> (Required): The path to your initial dataset (e.g., my_data.csv, sales_report.xlsx). DSolve will copy this file into your new project's data/raw directory.

<target_feature> (Required): The name of the column in your dataset that you intend to predict or analyze as your primary target.

--name <project_name> (Optional): A custom name for your project directory. If omitted, DSolve will automatically generate a name like dsolve_project_yourfilename.

--force (Optional): Use this flag if a directory with the chosen project_name already exists and you wish to overwrite its contents. Use with caution, as this will delete existing files!

What happens:

A new project directory is created with a standardized folder structure (data/raw, notebooks, scripts, reports, etc.).

A .dsolve hidden directory is created to mark the project root and store internal metadata.

A uv virtual environment is set up (unless --no-uv is specified during init if you run it separately).

Your provided data file is copied into the data/raw directory.

AI Analysis: DSolve's intelligent agents analyze your data and the specified target feature, providing:

Initial observations and feature characteristics.

Identification of potential challenges (missing values, outliers, etc.).

Suggestions for exploratory data analysis (EDA).

Initial thoughts on suitable machine learning models.

AI Solution Proposal: Based on the data analysis, the AI generates a detailed, step-by-step plan to tackle your data science problem, covering data preprocessing, feature engineering, model selection, training, evaluation, and more.

The data file is automatically registered in the project's manifest with an AI-generated description.

Initial Python code for data cleaning and a summary report are generated and saved.

Example:

dsolve solve "path/to/my_customer_data.csv" "churn_status" --name "customer_churn_prediction"

Option B: Create an Empty Project Structure (dsolve init)
If you prefer to set up an empty project structure first and add data later, use the init command.

Command Structure:

dsolve init <project_name> [--force] [--no-git] [--no-uv]

<project_name> (Required): The name of the new project directory.

--force (Optional): Overwrite an existing directory.

--no-git (Optional): Skip Git repository initialization.

--no-uv (Optional): Skip uv virtual environment setup.

Example:

dsolve init my_new_data_project

3. Next Steps After Project Creation
Once your project is created (either via solve or init):

Navigate into your project directory:

cd <your_project_name>

Activate your virtual environment:

Linux/macOS:

source .venv/bin/activate

Windows (PowerShell):

.venv\Scripts\activate

This ensures all project dependencies are isolated and managed correctly.

Explore the AI Output (if using solve): Review the analysis results and solution plan directly in your console. These insights guide your next steps.

Use other DSolve commands:

dsolve register <file_path> --type <type> [--auto-describe]: Add more files to your project's manifest.

dsolve list: See all registered files.

dsolve describe <file_path>: Get a detailed AI-generated description of a registered file.

dsolve explore <file_path>: Get AI-driven insights and anomaly detection for a data file.

dsolve analyze <file_path> "<analysis_request>": Perform conceptual statistical analysis.

dsolve hypothesize <file_path>: Generate testable statistical hypotheses.

dsolve suggest-command "<your_query>" [--file <file_path>]: Get CLI command suggestions from AI.

dsolve generate code "<task_description>" [-o <filename>] [--file <file_path>]: Generate Python code.

dsolve generate report "<report_topic>" [-o <filename>] [--context "<data_context>"]: Generate Markdown reports.

DSolve aims to be your intelligent partner throughout your data science journey, from initial setup to generating insights and code. Experiment with the commands and let the AI assist you!
        """
    console.print(guide_content, style="bold cyan")

if __name__ == "__main__":
    app()

