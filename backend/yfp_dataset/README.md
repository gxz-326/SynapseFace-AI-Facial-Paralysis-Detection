# YFP Facial Paralysis Dataset Setup

This directory contains the configuration for training and using the YFP (Yang Facial Palsy) dataset for facial paralysis detection.

## Dataset Structure

The YFP dataset should be organized as follows:

```
yfp_dataset/
├── normal/        # Normal/healthy face images
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── paralyzed/     # Paralyzed face images
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Training the Model

To train a new model using the YFP dataset:

1. Place your YFP dataset images in the appropriate directories:
   - Normal/healthy faces go in `yfp_dataset/normal/`
   - Paralyzed faces go in `yfp_dataset/paralyzed/`

2. Run the training script:
   ```bash
   cd backend
   python yfp_ensemble.py
   ```

3. The trained model will be saved as `yfp_voting_classifier.joblib`

## Model Files

- `yfp_voting_classifier.joblib` - Trained model using YFP dataset
- `voting_classifier.joblib` - Legacy model (old dataset)

## Switching Between Datasets

The system is currently configured to use the YFP dataset. The model.py file loads `yfp_voting_classifier.joblib`.

To switch back to the legacy dataset:
1. Change the model filename in `model.py` line 133 from `yfp_voting_classifier.joblib` to `voting_classifier.joblib`
2. Restart the Flask application

## Notes

- Supported image formats: .jpg, .jpeg, .png
- Images should contain clear frontal face views for optimal landmark detection
- The system uses dlib's 68-point facial landmark detection
- 28 symmetry features are extracted from the landmarks for classification