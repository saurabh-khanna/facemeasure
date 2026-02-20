FROM python:3.9-slim

WORKDIR /app

# Install system deps needed by py-feat/torch and healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Make feat resources directory writable and pre-download all models
# so the container never needs to write to site-packages at runtime.
RUN FEAT_RESOURCES=$(python -c "import feat, os; print(os.path.join(feat.__path__[0], 'resources'))") && \
    chmod -R a+rw "$FEAT_RESOURCES" && \
    python -c "\
from feat import Detector; \
Detector(face_model='retinaface', landmark_model='mobilefacenet', \
         au_model='xgb', emotion_model='svm', \
         facepose_model='img2pose', device='cpu')"

# Copy app code
COPY . .

# Appliku sets $PORT; default to 8501
ENV PORT=8501
EXPOSE ${PORT}

HEALTHCHECK CMD curl --fail http://localhost:${PORT}/_stcore/health || exit 1

CMD streamlit run home.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true
