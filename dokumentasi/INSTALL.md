# KODALY GESTURE RECOGNITION - INSTALLATION GUIDE

## Quick Install (Minimal)
```bash
pip install -r requirements-minimal.txt
```

## Full Install (With optional packages)
```bash
pip install -r requirements.txt
```

## Manual Install (Core packages)
```bash
pip install mediapipe opencv-python scikit-learn pandas numpy scipy
```

## Verify Installation
```bash
python -c "import mediapipe, cv2, sklearn, pandas, numpy, scipy; print('✅ All packages installed successfully!')"
```

## Run Application
```bash
# Train models (if needed)
python train_right_hand.py
python train_left_hand.py

# Run gesture recognition
python main_smart.py
```

## System Requirements
- Python 3.8+
- Camera/Webcam
- Windows/Linux/MacOS

## Notes
- MediaPipe requires camera access
- Models are pre-trained and saved in model_python/
- No Java required (pure Python implementation)
