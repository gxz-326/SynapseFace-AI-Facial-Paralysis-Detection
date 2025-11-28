import dlib
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score
import joblib
import os

detector = dlib.get_frontal_face_detector()
predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

def calculate_angle(point1, point2, point3):
    vector1 = point1 - point2
    vector3 = point3 - point2
    dot_product = np.dot(vector1, vector3)
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector3)

    if np.isclose(norm_product, 0.0):
        return 0.0

    angle_radians = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

def calculate_slope(point1, point2):
    if point2[0] == point1[0]:
        return 0.0
    else:
        return (point2[1] - point1[1]) / (point2[0] - point1[0])

def calculate_max_ratio(value1, value2):
    return max(value1 / value2, value2 / value1)

def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def process_landmarks(landmarks):
    key_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

    features = [
        calculate_angle(key_points[0], key_points[9], key_points[0]),  # f0
        calculate_angle(key_points[2], key_points[7], key_points[2]),  # f1
        calculate_angle(key_points[4], key_points[5], key_points[4]),  # f2
        max(key_points[2, 1] / key_points[7, 1], key_points[7, 1] / key_points[2, 1],
            key_points[4, 1] / key_points[5, 1], key_points[5, 1] / key_points[4, 1]),  # f3
        calculate_slope(key_points[0], key_points[9]),  # f4
        calculate_slope(key_points[2], key_points[7]),  # f5
        calculate_slope(key_points[4], key_points[5]),  # f6
        calculate_angle(key_points[10], key_points[19], key_points[10]),  # f7
        calculate_max_ratio(calculate_distance(key_points[11], key_points[13]),
                            calculate_distance(key_points[12], key_points[14])),  # f8
        calculate_max_ratio(calculate_distance(key_points[15], key_points[17]),
                            calculate_distance(key_points[16], key_points[18])),  # f9
        calculate_max_ratio(calculate_distance(key_points[19], key_points[21]),
                            calculate_distance(key_points[20], key_points[22])),  # f10
        calculate_max_ratio(calculate_distance(key_points[23], key_points[25]),
                            calculate_distance(key_points[24], key_points[26])),  # f11
        calculate_max_ratio(calculate_distance(key_points[23], key_points[37]),
                            calculate_distance(key_points[27], key_points[22])),  # f12
        calculate_max_ratio(calculate_distance(key_points[27], key_points[28]),
                            calculate_distance(key_points[29], key_points[30])),  # f13
        calculate_angle(key_points[28], key_points[34], key_points[28]),  # f14
        calculate_max_ratio(calculate_distance(key_points[31], key_points[32]),
                            calculate_distance(key_points[33], key_points[34])),  # f15
        calculate_max_ratio(calculate_distance(key_points[35], key_points[36]),
                            calculate_distance(key_points[37], key_points[38])),  # f16
        calculate_max_ratio(calculate_distance(key_points[39], key_points[40]),
                            calculate_distance(key_points[41], key_points[42])),  # f17
        calculate_max_ratio(calculate_distance(key_points[43], key_points[44]),
                            calculate_distance(key_points[45], key_points[46])),  # f18
        calculate_max_ratio(calculate_distance(key_points[48], key_points[51]),
                            calculate_distance(key_points[54], key_points[57])),  # f19
        calculate_max_ratio(calculate_distance(key_points[49], key_points[50]),
                            calculate_distance(key_points[55], key_points[56])),  # f20
        calculate_max_ratio(calculate_distance(key_points[52], key_points[53]),
                            calculate_distance(key_points[58], key_points[59])),  # f21
        calculate_angle(key_points[23], key_points[27], key_points[23]),  # f22
        calculate_angle(key_points[22], key_points[37], key_points[22]),  # f23
        calculate_max_ratio(calculate_distance(key_points[36], key_points[38]),
                            calculate_distance(key_points[37], key_points[38])),  # f24
        calculate_max_ratio(calculate_distance(key_points[52], key_points[60]),
                            calculate_distance(key_points[53], key_points[61])),  # f25
        calculate_max_ratio(calculate_distance(key_points[50], key_points[51]),
                            calculate_distance(key_points[58], key_points[59])),  # f26
        calculate_distance(key_points[46], key_points[23]) / calculate_distance(key_points[43], key_points[23]),  # f27
        calculate_distance(key_points[51], key_points[23]) / calculate_distance(key_points[48], key_points[23]),  # f28
    ]

    return features[:28]

def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    features = np.zeros(28)

    if faces:
        landmarks = predictor(gray, faces[0])
        features = process_landmarks(landmarks)

    return features

def train_and_save_models(data_paths, labels):
    X_train = []
    X_train_models = []
    models = []

    for image_path in data_paths:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        print(f"Processing {image_path}")
        
        features = extract_features(img)
        X_train.append(features)

    if len(X_train) == 0:
        raise ValueError("No valid images found in the dataset")

    X_train = np.vstack(X_train)
    X_train_models.append(X_train)
    y_train = labels[:len(X_train)]  # Ensure labels match the number of processed images

    # Train the SVM model
    svm_model = SVC(kernel='rbf', C=10000, gamma=0.001, probability=True)
    svm_model.fit(X_train, y_train)
    models.append(('svm', svm_model))

    # Train a Random Forest model
    random_forest = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    random_forest.fit(X_train, y_train)
    models.append(('random_forest', random_forest))

    # Train a Gradient Boosting model
    gradient_boosting = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gradient_boosting.fit(X_train, y_train)
    models.append(('gradient_boosting', gradient_boosting))

    return models, X_train_models

# Load YFP dataset
yfp_dataset_path = 'yfp_dataset'
data_paths = []
labels = []

# YFP dataset typically has normal and paralyzed folders
normal_path = os.path.join(yfp_dataset_path, 'normal')
paralyzed_path = os.path.join(yfp_dataset_path, 'paralyzed')

# Load normal images (label 0)
if os.path.exists(normal_path):
    for filename in os.listdir(normal_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            data_paths.append(os.path.join(normal_path, filename))
            labels.append(0)

# Load paralyzed images (label 1)
if os.path.exists(paralyzed_path):
    for filename in os.listdir(paralyzed_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            data_paths.append(os.path.join(paralyzed_path, filename))
            labels.append(1)

if len(data_paths) == 0:
    print("YFP dataset not found. Creating placeholder structure.")
    # Create placeholder directories and sample data paths
    os.makedirs(normal_path, exist_ok=True)
    os.makedirs(paralyzed_path, exist_ok=True)
    print(f"Please add YFP dataset images to:")
    print(f"  - {normal_path} (normal/healthy faces)")
    print(f"  - {paralyzed_path} (paralyzed faces)")
    exit()

print(f"Found {len(data_paths)} images in YFP dataset")
print(f"Normal images: {labels.count(0)}")
print(f"Paralyzed images: {labels.count(1)}")

labels = np.array(labels)

# Split the dataset into training and testing sets
X_train_paths, X_test_paths, y_train, y_test = train_test_split(
    data_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

# Train the models and save them to files
models, X_train_models = train_and_save_models(X_train_paths, y_train)

# Create a Voting Classifier
voting_classifier = VotingClassifier(estimators=models, voting='soft')

# Train the Voting Classifier
voting_classifier.fit(X_train_models[0], y_train)

# Save the trained Voting Classifier to a file
joblib.dump(voting_classifier, 'yfp_voting_classifier.joblib')
print("YFP model saved as 'yfp_voting_classifier.joblib'")

# Load the trained Voting Classifier
ensemble_model = joblib.load('yfp_voting_classifier.joblib')

# Iterate through the test images and make predictions using the ensemble model
ensemble_predictions = []

for test_path in X_test_paths:
    test_img = cv2.imread(test_path)
    if test_img is not None:
        ensemble_prediction = ensemble_model.predict([extract_features(test_img)])
        ensemble_predictions.append(ensemble_prediction[0])

# Convert ensemble predictions to a numpy array for consistency
ensemble_predictions = np.array(ensemble_predictions)

# Calculate accuracy, F1 score, confusion matrix, and recall for the ensemble model
ensemble_accuracy = accuracy_score(y_test[:len(ensemble_predictions)], ensemble_predictions)
ensemble_f1 = f1_score(y_test[:len(ensemble_predictions)], ensemble_predictions)
ensemble_conf_matrix = confusion_matrix(y_test[:len(ensemble_predictions)], ensemble_predictions)
ensemble_recall = recall_score(y_test[:len(ensemble_predictions)], ensemble_predictions)
ensemble_precision = precision_score(y_test[:len(ensemble_predictions)], ensemble_predictions)

print(f"YFP Ensemble Accuracy: {ensemble_accuracy * 100:.2f}%")
print(f"YFP Ensemble F1 Score: {ensemble_f1:.2f}")
print(f"YFP Ensemble Confusion Matrix:\n{ensemble_conf_matrix}")
print(f"YFP Ensemble Recall: {ensemble_recall:.2f}")
print(f"YFP Ensemble Precision: {ensemble_precision:.2f}")