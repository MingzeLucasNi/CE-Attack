# Cross-Entropy Attacking to NLP Models via Rare-Even Simulation
This repository contains the implementation of a Cross-Entropy Attack (CEA) on various NLP models for the paper: Cross-Entropy Attacking to NLP Models via Rare-Even Simulation for AAAI2025. The attack is designed to generate adversarial examples that can fool text classifiers and neural machine translation (NMT) models. The code is modular, allowing for easy integration with different datasets and models from the Hugging Face library.

## Abstract
Black-box textual adversarial attacks are challenging due to the lack of model information and the discrete, non-differentiable nature of text. Existing methods often lack versatility for attacking different models, suffer from limited attacking performance due to the inefficient optimization with word saliency ranking, and frequently sacrifice semantic integrity to achieve better attack outcomes. This paper introduces a novel approach to textual adversarial attacks, which we call Cross-Entropy Attacks (CEA), that uses Cross-Entropy optimization to address the above issues. Our CEA approach defines adversarial objectives for both soft-label and hard-label settings and employs CE optimization to identify optimal replacements. Through extensive experiments on document classification and language translation problems, we demonstrate that our attack method excels in terms of attacking performance, imperceptibility, and sentence quality.


## Table of Contents

- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [Running the Attack](#running-the-attack)
  - [Command-Line Arguments](#command-line-arguments)
- [Customization](#customization)

## Project Structure

```
.
├── README.md
├── config.py
├── dataset_preparation.py
├── uti.py
├── attack.py
├── main.py
└── requirements.txt
```

- **`config.py`**: Handles command-line arguments for the attack script.
- **`dataset_preparation.py`**: Contains functions for loading datasets and models from Hugging Face.
- **`uti.py`**: Contains utility functions used in the attack, such as generating substitution sets and calculating similarity scores.
- **`attack.py`**: Implements the Cross-Entropy Attack, including performance evaluation methods.
- **`main.py`**: The main script for running the attack.
- **`requirements.txt`**: Lists the Python dependencies required for the project.

## Setup and Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Installation Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/cea-nlp-attack.git
   cd cea-nlp-attack
   ```

2. **Install the Required Python Packages**

   Install the required dependencies using pip:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file includes:

   - `transformers`
   - `datasets`
   - `sentence-transformers`
   - `rouge-score`
   - `nltk`
   - `OpenHowNet`

3. **Download NLTK Data**

   The script uses NLTK for tokenization. You need to download the required NLTK data:

   ```python
   python -m nltk.downloader punkt
   ```

## Usage

### Soft-Label Attack on SST2

```bash
python main.py --text "This movie was absolutely amazing, with stellar performances and a gripping plot." --true_label "POSITIVE" --dataset "sst2" --model_type "classifier" --attack_type "soft"
```

### Hard-Label Attack on AG News

```bash
python main.py --text "The stock market crashed today, causing widespread panic." --true_label "Business" --dataset "ag_news" --model_type "classifier" --attack_type "hard"
```

### NMT Attack on WMT Task 1 (English to French)

```bash
python main.py --text "The quick brown fox jumps over the lazy dog." --true_label None --dataset "wmt1" --model_type "nmt" --attack_type "soft"
```

### Command-Line Arguments

The following command-line arguments can be used to customize the attack:

- **`--text`**: The input text you want to attack.
- **`--true_label`**: The true label of the text (for classifiers).
- **`--dataset`**: The dataset name. Choose from `"ag_news"`, `"imdb"`, `"sst2"` for classifiers, or `"wmt1"`, `"wmt2"` for NMT tasks.
- **`--model_type`**: The type of model to attack. Choose `"classifier"` or `"nmt"`.
- **`--attack_type`**: The type of attack to perform. Choose `"soft"` for soft-label attacks or `"hard"` for hard-label attacks.
- **`--max_iterations`**: (Optional) Maximum iterations for the attack (default: 100).
- **`--num_candidates`**: (Optional) Number of candidate substitutions to generate in each iteration (default: 10).
- **`--rho`**: (Optional) Proportion of top candidates to retain for updating the attack (default: 0.2).

### Important Notice for Users

Since this project allows for the use of different victim models and datasets, users may need to adjust the `dataset_preparation.py` file to accommodate their specific models and datasets. The provided code supports commonly used models and datasets, but for custom scenarios, you might need to modify the model loading and dataset preparation functions accordingly.

## Customization

### Adding New Datasets or Models

To add a new dataset or model, modify the `load_model` and `load_dataset` functions in the `dataset_preparation.py` file. You can follow the existing structure to include new datasets and models.

### Extending the Attack

The `CrossEntropyAttack` class in `attack.py` is designed to be modular. You can extend or modify the attack by altering the substitution generation or performance evaluation methods.

---

This `README.md` provides a concise guide for users to understand, install, and run the Cross-Entropy Attack on various NLP models, with clear instructions on how to adjust the `dataset_preparation.py` file for different models and datasets.