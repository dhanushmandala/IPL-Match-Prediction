# IPL Match Winner Prediction

A machine-learning project that predicts the likely winner of an Indian Premier League (IPL) match using match details such as the two teams, venue, city, toss decision, and whether Team 1 won the toss.

The project uses a **Decision Tree Classifier** trained on historical IPL match data and provides an interactive prediction interface built with **Streamlit**.

## Features

- Select Team 1 and Team 2 from the available historical team list.
- Select the match venue and city.
- Choose the toss decision (`bat` or `field`).
- Indicate whether Team 1 won the toss.
- Predict which of the two selected teams is more likely to win.
- Load a previously trained model and its categorical encoders from a Joblib artifact.

## Project Structure

```text
ipl/
├── app.py
├── ipl_matches.csv
├── ipl_decision_tree_pipeline.pkl
└── project---4 (1).ipynb
```

| File | Description |
|---|---|
| `app.py` | Streamlit application for entering match details and displaying a prediction. |
| `ipl_matches.csv` | Historical IPL match dataset used during model development. |
| `ipl_decision_tree_pipeline.pkl` | Saved model artifact containing the trained classifier, label encoders, scaler, and feature list. |
| `project---4 (1).ipynb` | Jupyter Notebook containing data exploration, preprocessing, model training, evaluation, and model export. |

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook
- Matplotlib and Seaborn (for notebook visualizations)

## Model Overview

The notebook prepares the target variable as follows:

- `1` — Team 1 won the match.
- `0` — Team 1 did not win (Team 2 won).

The model uses these six input features:

1. `team1`
2. `team2`
3. `venue`
4. `city`
5. `toss_decision`
6. `team1_toss_winner`

Categorical fields are converted using fitted `LabelEncoder` objects. The notebook also applies `StandardScaler`, splits the data into training and test sets using an 80:20 split, and tunes a `DecisionTreeClassifier` with `GridSearchCV` and stratified 5-fold cross-validation.

**Note:** Decision trees do not generally require feature scaling. The deployed `app.py` currently encodes the categorical values and predicts directly; it does not apply the saved scaler. If the model was trained on scaled features, update the app to apply `artifacts["scaler"].transform(...)` before prediction so training and inference use the same preprocessing.

## Setup and Installation

### 1. Install Python

Use Python 3.9 or later. Confirm that Python and pip are available:

```bash
python --version
pip --version
```

### 2. Open the project folder

In a terminal, change into the folder containing `app.py`, the CSV file, and the model file:

```bash
cd ipl
```

### 3. Create and activate a virtual environment (recommended)

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install streamlit pandas scikit-learn joblib
```

For running the notebook and its visualizations, also install:

```bash
pip install jupyter notebook matplotlib seaborn
```

## Run the Streamlit Application

From the `ipl` folder, run:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal, usually:

```text
http://localhost:8501
```

Open that address in your browser. Select the match details and click **Predict** to see the predicted winner.

## Run the Jupyter Notebook

To inspect the analysis or retrain the model:

```bash
jupyter notebook
```

Open `project---4 (1).ipynb` and run the cells in order. The notebook exports the trained artifacts to:

```text
ipl_decision_tree_pipeline.pkl
```

Keep the generated `.pkl` file in the same directory as `app.py`.

## Important Notes and Limitations

- The prediction is an estimate based on patterns in the historical dataset; it is not a guarantee of the actual match result.
- The available dropdown options are taken from the encoders saved in the model artifact. New teams, venues, or cities are not automatically supported.
- The current app does not prevent selecting the same team for both Team 1 and Team 2.
- The current app uses the selected toss information but does not include player availability, playing XI, injuries, current team form, weather, or live match conditions.
- The dataset contains 1,243 rows and 10 columns. Actual training rows may be fewer after removing records without a winner.
- Do not expose or load model files from untrusted sources. Joblib/pickle files can execute code when loaded.

## Troubleshooting

**`FileNotFoundError: ipl_decision_tree_pipeline.pkl`**

Run the app from the `ipl` project directory and confirm that the `.pkl` file is present beside `app.py`.

**`ModuleNotFoundError`**

Activate the virtual environment and install the dependencies listed above.

**`streamlit` is not recognized**

Run Streamlit through Python:

```bash
python -m streamlit run app.py
```

**An unseen category causes a label-encoding error**

The app can only encode categories known to the saved encoders. Retrain the model with the new category or add an explicit validation message before encoding.

## Future Improvements

- Apply the saved scaler consistently during prediction, if the trained model expects scaled inputs.
- Validate that Team 1 and Team 2 are different.
- Add clear handling for unknown categories and missing values.
- Display prediction probabilities and model evaluation metrics.
- Improve the model evaluation with additional baselines and time-aware validation.
- Add more informative match features, while avoiding data leakage.
