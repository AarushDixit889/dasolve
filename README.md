# Creating a dasolve Project: A Step-by-Step Guide

> **dasolve** is a powerful command-line interface designed to streamline your data science workflows by automating project setup, data analysis, and solution planning with the help of AI.

## 📋 Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed on your system
- **uv** (recommended for virtual environment and package management) or **pip** installed

## 🚀 1. Installing dasolve

### Option 1: Install dasolve from PyPI

```bash
pip install dasolve
```

### Option 2: Install dasolve from GitHub

```bash
git clone https://github.com/aarushdixit889/dasolve.git
```

```bash
cd dasolve
```

```bash
pip install -e .
```


Once installed, you should be able to run:

```bash
dasolve --help
```

This will show you all the available commands.

## 📁 2. Creating a New Project

dasolve offers two primary ways to start a new project:

### Option A: Create a Project with Initial Data and AI Analysis (`dasolve solve`) - **Recommended**

This is the most comprehensive way to start, as it not only sets up your project but also immediately processes your data with AI to provide initial analysis and a solution plan.

#### Command Structure:

```bash
dasolve solve <file_path> <target_feature> [--name <project_name>] [--force]
```

**Parameters:**
- `<file_path>` **(Required)**: The path to your initial dataset (e.g., `my_data.csv`, `sales_report.xlsx`). dasolve will copy this file into your new project's `data/raw` directory.
- `<target_feature>` **(Required)**: The name of the column in your dataset that you intend to predict or analyze as your primary target.
- `--name <project_name>` **(Optional)**: A custom name for your project directory. If omitted, dasolve will automatically generate a name like `dasolve_project_yourfilename`.
- `--force` **(Optional)**: Use this flag if a directory with the chosen `project_name` already exists and you wish to overwrite its contents. ⚠️ **Use with caution, as this will delete existing files!**

#### What happens:

1. **Project Structure**: A new project directory is created with a standardized folder structure:
   ```
   project_name/
   ├── data/
   │   ├── raw/
   │   ├── processed/
   │   └── external/
   ├── notebooks/
   ├── scripts/
   ├── reports/
   ├── models/
   └── .dasolve/
   ```

2. **Environment Setup**: A `uv` virtual environment is set up (unless `--no-uv` is specified during init if you run it separately).

3. **Data Processing**: Your provided data file is copied into the `data/raw` directory.

4. **AI Analysis**: dasolve's intelligent agents analyze your data and the specified target feature, providing:
   - Initial observations and feature characteristics
   - Identification of potential challenges (missing values, outliers, etc.)
   - Suggestions for exploratory data analysis (EDA)
   - Initial thoughts on suitable machine learning models

5. **AI Solution Proposal**: Based on the data analysis, the AI generates a detailed, step-by-step plan to tackle your data science problem, covering:
   - Data preprocessing
   - Feature engineering
   - Model selection
   - Training
   - Evaluation
   - And more!

6. **Code Generation**: Initial Python code for data cleaning and a summary report are generated and saved.

#### Example:

```bash
dasolve solve "path/to/my_customer_data.csv" "churn_status" --name "customer_churn_prediction"
```

### Option B: Create an Empty Project Structure (`dasolve init`)

If you prefer to set up an empty project structure first and add data later, use the `init` command.

#### Command Structure:

```bash
dasolve init <project_name> [--force] [--no-git] [--no-uv]
```

**Parameters:**
- `<project_name>` **(Required)**: The name of the new project directory.
- `--force` **(Optional)**: Overwrite an existing directory.
- `--no-git` **(Optional)**: Skip Git repository initialization.
- `--no-uv` **(Optional)**: Skip `uv` virtual environment setup.

#### Example:

```bash
dasolve init my_new_data_project
```

## 🔄 3. Next Steps After Project Creation

Once your project is created (either via `solve` or `init`):

### Navigate into your project directory:

```bash
cd <your_project_name>
```

### Activate your virtual environment:

**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\activate
```

> **💡 Tip**: This ensures all project dependencies are isolated and managed correctly.

### Explore the AI Output (if using `solve`)

Review the analysis results and solution plan directly in your console. These insights guide your next steps.

### Use other dasolve commands:

| Command | Description |
|---------|-------------|
| `dasolve register <file_path> --type <type> [--auto-describe]` | Add more files to your project's manifest |
| `dasolve list` | See all registered files |
| `dasolve describe <file_path>` | Get a detailed AI-generated description of a registered file |
| `dasolve explore <file_path>` | Get AI-driven insights and anomaly detection for a data file |
| `dasolve analyze <file_path> "<analysis_request>"` | Perform conceptual statistical analysis |
| `dasolve hypothesize <file_path>` | Generate testable statistical hypotheses |
| `dasolve suggest-command "<your_query>" [--file <file_path>]` | Get CLI command suggestions from AI |
| `dasolve generate code "<task_description>" [-o <filename>] [--file <file_path>]` | Generate Python code |
| `dasolve generate report "<report_topic>" [-o <filename>] [--context "<data_context>"]` | Generate Markdown reports |

## 🎯 Example Workflow

Here's a typical workflow using dasolve:

```bash
# 1. Create a new project with data
dasolve solve "customer_data.csv" "purchase_amount" --name "customer_analysis"

# 2. Navigate to project
cd customer_analysis

# 3. Activate environment
source .venv/bin/activate

# 4. Explore your data
dasolve explore "data/raw/customer_data.csv"

# 5. Generate some analysis code
dasolve generate code "Create a correlation matrix for numerical features" --file "data/raw/customer_data.csv"

# 6. Generate a report
dasolve generate report "Customer Segmentation Analysis" --context "Analyzing customer purchase patterns"
```

## 🚀 Getting Started Tips

> **💡 Pro Tip**: Start with the `dasolve solve` command for the most comprehensive experience. It will give you immediate insights and a clear roadmap for your data science project.

> **🔧 Best Practice**: Always review the AI-generated analysis and solution plan before proceeding. While AI provides excellent guidance, domain knowledge is still crucial.

> **📊 Data Quality**: Ensure your data is in a common format (CSV, Excel, JSON) and has clear column names for the best AI analysis results.

---

**dasolve** aims to be your intelligent partner throughout your data science journey, from initial setup to generating insights and code. Experiment with the commands and let the AI assist you! 🎉