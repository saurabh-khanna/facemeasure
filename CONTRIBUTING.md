# Contributing to facemeasure

Thank you for your interest in contributing to **facemeasure**! This document provides guidelines and instructions for contributing.

## How to Contribute

### Reporting Bugs

If you find a bug, please [open an issue](https://github.com/saurabh-khanna/facemeasure/issues/new) with:

- A clear, descriptive title.
- Steps to reproduce the problem.
- Expected vs. actual behaviour.
- Your environment (OS, Python version, browser).

### Suggesting Features

Feature requests are welcome. Please [open an issue](https://github.com/saurabh-khanna/facemeasure/issues/new) describing:

- The use case or research workflow the feature would support.
- How you envision it working.

### Submitting Changes

1. **Fork** the repository and create a branch from `main`.
2. **Install dependencies** locally:
   ```bash
   pip install -r requirements.txt
   pip install pytest
   ```
3. **Make your changes** — keep commits focused and well-described.
4. **Run the tests** to make sure nothing is broken:
   ```bash
   pytest test_pipeline.py -v
   ```
5. **Open a pull request** against `main` with a clear description of what you changed and why.

### Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- Use descriptive variable names and add docstrings to functions.
- Keep the single-file architecture (`home.py`) unless there is a strong reason to refactor.

### Testing

- Add or update tests in `test_pipeline.py` for any new or changed functionality.
- Tests should run without requiring a GPU or network access (use synthetic data where possible).

## Code of Conduct

All contributors are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Please be respectful and constructive in all interactions.

## Questions?

If you have questions about contributing, feel free to [open a discussion](https://github.com/saurabh-khanna/facemeasure/issues) or reach out via the repository's issue tracker.
