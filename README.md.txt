# Email Spam Detection Project

## What it does
Classifies emails as Spam or Ham using Machine Learning.

## Dataset
- File: spam.csv
- Columns: v1 = label (spam/ham), v2 = email text

## How to run
1. Install libraries: pip install -r requirements.txt
2. Run program: python main.py

## Recent Fix (Dec 31, 2025)
- Cause: The script originally balanced the dataset after the train/test split, so the model trained on the full imbalanced dataset (mostly `ham`) and learned to predict the majority class.
- Fix: Balancing is now performed before `train_test_split` in `main_code.py`, so the model trains on balanced classes and no longer predicts `ham` for most inputs.

## Usage example
- Train and test interactively:
	```powershell
	python "main.py/main_code.py"
	```
- Quick non-interactive test (pipes a sample email to stdin):
	```powershell
	echo "You won a free prize claim now" | python "main.py/main_code.py"
	```

## Notes
- Model and vectorizer are saved as `spam_model.pkl` and `vectorizer.pkl` after training.
- If you want to retrain on the full dataset without sampling, remove the sampling line in `main_code.py`.

## Author
Dharshini A
