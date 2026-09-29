# Large Language Models as Clinical Data Synthesizers

Code and data for the article

> Kara K, Gunel T. *Large Language Models as Clinical Data Synthesizers*. International Journal of Data Science and Analytics (under review).

The study asks whether general-purpose large language models (LLMs) can generate useful synthetic clinical cohorts when they are given nothing but the univariate summary of a real dataset. Four widely used LLMs received the same prompt, each produced three independent cohorts, and the twelve cohorts were evaluated for fidelity, memorization of real records, disclosure risk and predictive performance on real patients.

## Repository contents

```
├── data/
│   ├── README.md                  data sources and variable coding
│   └── synthetic/                 the 12 generated cohorts (4 models × 3 runs)
├── prompt/
│   └── prompt.txt                 exact prompt submitted to every model
├── run_analysis.py                complete analysis; writes all tables to results/tables
├── make_figures.py                main and supplementary figures; writes to figures/
├── requirements.txt
└── LICENSE
```

## Data

**Reference data.** The Heart Disease Dataset (Comprehensive) combines five cohorts of the UCI Machine Learning Repository (Cleveland, Hungarian, Switzerland, Long Beach VA and Statlog) and contains 1,190 records. It is not redistributed here. Download it from IEEE Dataport (https://doi.org/10.21227/dz4t-cm36) and save it as `data/real_data.csv`; the column names must be those listed in `data/README.md`.

**Synthetic cohorts.** All cohorts were generated on 29 September 2026 through the public web interfaces, each in a new conversation with the prompt in `prompt/prompt.txt`:

| Files | Service | Model and setting |
| --- | --- | --- |
| `chatgpt_run1-3.csv` | ChatGPT | GPT-5.6 Luna, Think enabled |
| `claude_run1-3.csv` | Claude | Claude Sonnet 5.5, medium effort |
| `deepseek_run1-3.csv` | DeepSeek | DeepSeek-V4.1-Flash, DeepThink enabled |
| `gemini_run1-3.csv` | Gemini | Gemini 3.1 Pro, extended thinking |

The values are exactly as returned by the models; only the header row was harmonized. Each file contains 200 records, except `gemini_run1.csv` (192 records).

## Reproducing the results

Python 3.10 or later is required. The results in the article were produced with Python 3.12 and the package versions pinned in `requirements.txt`.

```bash
pip install -r requirements.txt
python run_analysis.py
python make_figures.py
```

`run_analysis.py --jobs N` fits models in parallel, and `run_analysis.py --quick` runs the whole pipeline with reduced settings as a functional check (its numbers differ from those in the article). Intermediate results are cached in `results/cache/`; an interrupted run resumes where it stopped, and deleting the folder forces a complete recomputation. `make_figures.py --dpi 600` produces figures at higher resolution.

All random processes use fixed seeds. With the pinned package versions the scripts reproduce every number in the article; other versions may cause small differences in the last decimal places.

## Citation

If you use the code or the synthetic cohorts, please cite the article (The link to the article will be added here once it is published) and the reference dataset:

- Siddhartha M. Heart disease dataset (comprehensive). IEEE Dataport (2020). https://doi.org/10.21227/dz4t-cm36
- Janosi A, Steinbrunn W, Pfisterer M, Detrano R. Heart disease. UCI Machine Learning Repository (1988). https://doi.org/10.24432/C52P4X

## License

The code is released under the MIT License. The synthetic cohorts are released under the Creative Commons Attribution 4.0 International License (CC BY 4.0); see `data/README.md`.
