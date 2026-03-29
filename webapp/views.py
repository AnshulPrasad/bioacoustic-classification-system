import os
import sys
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings

# --- CRITICAL PATH FIX ---
# This tells Python to look inside your 'src' folder for imports,
# completely fixing the "ModuleNotFoundError" you saw earlier.
SRC_DIR = os.path.join(settings.BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# Now we can safely import your ML code!
from src.predict import Predictor

from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="AnshulPrasad/avian-vocal-classification-system",  # your model repo
    filename="best_model.pth"
)

# Initialize the model globally (using absolute paths to be safe)
PREDICTOR = Predictor(
    model_path=os.path.join(settings.BASE_DIR, "models", "checkpoints", "best_model.pth"),
    mapping_path=os.path.join(settings.BASE_DIR, "models", "class_mapping.json")
)


def upload_and_predict(request):
    if request.method == 'POST' and request.FILES.get('audio_file'):
        uploaded_file = request.FILES['audio_file']

        # 1. Save the file temporarily
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)

        try:
            # 2. Run the REAL prediction
            predicted_label, confidence = PREDICTOR.predict(file_path)

            # 3. Prepare data to send back to the frontend
            context = {
                'label': predicted_label.title(),
                'confidence': f"{confidence:.2%}",
                'filename': uploaded_file.name
            }
            return render(request, 'result.html', context)

        except Exception as e:
            return render(request, 'index.html', {'error': f"Error processing audio: {str(e)}"})

        finally:
            # 4. Delete the audio file to save disk space
            if os.path.exists(file_path):
                os.remove(file_path)

    # If it's a GET request, just show the upload form
    return render(request, 'index.html')