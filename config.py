"""Central configuration for FinGuard AI (paths, constants, palette, risk thresholds)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "data.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ASSETS_DIR = ROOT / "assets"
HISTORY_DB = REPORTS_DIR / "prediction_history.db"
SAMPLE_CSV = ROOT / "data" / "sample_input.csv"
TEMPLATE_CSV = ROOT / "data" / "input_template.csv"

TARGET = "Bankrupt?"
CONSTANT_FLAG = "Net Income Flag"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
DUPLICATE_FEATURE_CORR = 0.9999   # |r| at/above this => perfectly redundant feature
LEAKAGE_AUC_FLAG = 0.95           # single-feature AUC above this is "suspiciously predictive"
WINSOR_LOW, WINSOR_HIGH = 0.01, 0.99

# Model registry: key -> display name + artifact file
MODELS = {
    "svm": {"name": "Support Vector Machine", "short": "SVM", "file": "svm_model.pkl"},
    "extra_trees": {"name": "Extra Trees Classifier", "short": "Extra Trees", "file": "extra_trees_model.pkl"},
    "catboost": {"name": "CatBoost Classifier", "short": "CatBoost", "file": "catboost_model.cbm"},
}
BEST_MODEL_FILE = MODELS_DIR / "best_model.pkl"
PREPROCESSING_FILE = MODELS_DIR / "preprocessing.pkl"
EVALUATION_FILE = REPORTS_DIR / "evaluation_results.json"
EDA_FILE = REPORTS_DIR / "eda_summary.json"
IMPORTANCE_FILE = REPORTS_DIR / "feature_importance.csv"
TEST_PRED_FILE = REPORTS_DIR / "test_predictions.csv"

# Risk bands are anchored to each model's decision threshold t (see src/prediction.py):
#   Low < t/2 <= Moderate < t <= High < (1+t)/2 <= Critical
RISK_LEVELS = ["Low", "Moderate", "High", "Critical"]

PALETTE = {
    "navy": "#0A1128", "charcoal": "#141B2D", "panel": "#111A33", "blue": "#3B82F6",
    "cyan": "#22D3EE", "purple": "#8B5CF6", "red": "#F43F5E", "green": "#10B981",
    "amber": "#F59E0B", "muted": "#8FA0BF",
}
RISK_COLORS = {"Low": "#10B981", "Moderate": "#F59E0B", "High": "#F97316", "Critical": "#F43F5E"}
MODEL_COLORS = {"svm": "#22D3EE", "extra_trees": "#8B5CF6", "catboost": "#3B82F6"}

DISCLAIMER = "FinGuard AI is a machine-learning project."
