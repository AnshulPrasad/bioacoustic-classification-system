from pathlib import Path

SPECIES_LIST = [
    ["Pycnonotus cafer", "Red-vented Bulbul"],
    ["Pycnonotus jocosus", "Red-whiskered Bulbul"],
    ["Athene brama", "Spotted Owlet"],
    ["Corvus splendens", "House Crow"],
    ["Dicrurus macrocercus", "Black Drongo"],
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPECTROGRAM_DIR = PROJECT_ROOT / "data" / "spectrograms"
SPLIT_DIR = PROJECT_ROOT / 'data" / "split'
MODEL_PATH = PROJECT_ROOT / "models" / "checkpoints" / "best_model.pth"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "outputs" / "confusion_matrix.png"
SPLIT_JSON_PATH = PROJECT_ROOT / "models" / "split_index.json"
CLASS_MAPPING_JSON = PROJECT_ROOT / "models" / "class_mapping.json"

for p in [RAW_DIR, PROCESSED_DIR, SPECTROGRAM_DIR, SPLIT_DIR, MODEL_PATH, CONFUSION_MATRIX_PATH, SPLIT_JSON_PATH, CLASS_MAPPING_JSON]:
    p.mkdir(parents=True, exist_ok=True)