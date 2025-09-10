# Install dependencies
pip install -r requirements.txt

# See EDA
check notebooks/eda.ipynb

# Train model
python src/train.py

# Run inference
python src/main.py --inference --context "Your context here" --question "Your question here"