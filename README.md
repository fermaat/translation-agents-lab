# Multilingual Translation Agents Research

Welcome to the Multilingual Translation Agents Research repository. This project explores advanced, multi-agent AI workflows for translation, evaluation, and research on large language models (LLMs) and translation quality.

## Overview

This repository demonstrates how to build, evaluate, and extend translation pipelines using state-of-the-art LLMs and agent-based architectures. It leverages [crewAI](https://crewai.com) for multi-agent coordination, and includes custom workflows, prompt templates, and evaluation tools for translation research.

Key features:
- **Multi-agent orchestration** for translation and quality assessment
- **Prompt engineering** for baseline and advanced translation tasks
- **Automated evaluation** of translation outputs
- **Extensible workflows** for research and experimentation
- **Integrated promptflow tracing** for workflow instrumentation and experiment tracking

## Repository Structure
```bash
. 
├── .env 
├── README.md 
├── working flow.py 
├── data/ 
├── prompts/ 
│ └── translation/ 
├── src/ │ 
│ ├── translation/ 
│ └── utils/ 
└── crewai_projects/ 
    └── test_project/
```


- **data/**: Contains test datasets and translation memories.
- **prompts/**: Prompt templates for translation agents and quality checkers.
- **research/**: Jupyter notebooks for experimental workflows and group chat simulations.
- **src/**: Core source code for translation workflows, evaluation, and utilities.
- **crewai_projects/**: Example crewAI projects, including a ready-to-run test project.

## Getting Started

### Prerequisites

- Python >=3.10, <=3.13
- [UV](https://docs.astral.sh/uv/) for dependency management
- [crewAI](https://crewai.com) for agent orchestration

### Installation

1. Install UV:
```bash
pip install uv
```
2. Navigate to a crewAI project (e.g., `crewai_projects/test_project`) and install dependencies:
```bash
crewai install
```
3. Add your `OPENAI_API_KEY` to the `.env` file.

### Running Example Project

To launch the example multi-agent workflow:

```bash
cd crewai_projects/test_project
crewai run
```

This will assemble a team of translation agents and generate a research report.

## Customization
- Edit src/translation/ to modify or extend translation workflows.
- Update prompt templates in prompts/translation/ for new agent behaviors.
- Add datasets to data/ for evaluation.
- Configure agents and tasks in crewai_projects/test_project/src/test_project/config/.

## Research Directions
- Experiment with new prompt strategies and agent roles.
- Evaluate translation quality using custom metrics.
- Integrate additional LLMs or external APIs.

## License
This repository is for research and portfolio demonstration purposes.

## Acknowledgments
crewAI for multi-agent orchestration
OpenAI for LLM APIs

📬 Contact
For questions, collaborations, or feedback, feel free to reach out:

📧 Email: fermaat.vl@gmail.com
🧑‍💻 GitHub: [@fermaat](https://github.com/fermaat)
🌐 [Website](https://fermaat.github.io)