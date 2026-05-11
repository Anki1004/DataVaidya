# Contributing to DataVaidya

Thanks for your interest in improving DataVaidya. This is a small note to get you started.

## Getting set up

1. Fork and clone the repository.
2. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate      # macOS / Linux
   pip install -r requirements.txt
   ```
3. Run the test suite before making changes:
   ```
   pytest
   ```

## Making changes

- Keep pull requests focused — one logical change per PR.
- Match the existing code style (type hints, docstrings on public functions).
- Add or update tests under `tests/` for any behaviour you change.
- If you touch profiling, cleaning, PII, validation, or exports, please ensure the relevant `tests/test_*.py` module still passes.

## Reporting issues

When filing a bug, please include:
- A minimal dataframe or file that reproduces the issue (with any PII removed).
- The full traceback.
- Your Python and pandas versions.

## Code of conduct

Be kind. Assume good intent. Review feedback is about the code, not the person.

## License

By contributing, you agree that your contributions will be licensed under the MIT License (see `LICENSE`).
