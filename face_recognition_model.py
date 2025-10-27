import os
import cv2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import mediapipe as mp
from typing import Optional, Tuple, Callable
import io

MODEL_PATH = "face_model.pkl"
DATASET_DIR = "dataset"

os.makedirs(DATASET_DIR, exist_ok=True)

mp_face_detection = mp.solutions.face_detection

def crop_face_and_embed(bgr_image: np.ndarray, detection) -> Optional[np.ndarray]:
    """Extract face from image and create embedding"""
    h, w = bgr_image.shape[:2]
    bbox = detection.location_data.relative_bounding_box
    
    x1 = int(max(0, bbox.xmin * w))
    y1 = int(max(0, bbox.ymin * h))
    x2 = int(min(w, (bbox.xmin + bbox.width) * w))
    y2 = int(min(h, (bbox.ymin + bbox.height) * h))
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    face = bgr_image[y1:y2, x1:x2]
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (64, 64), interpolation=cv2.INTER_AREA)
    embedding = face_resized.flatten().astype(np.float32) / 255.0
    
    return embedding

def extract_face_embedding(image: np.ndarray) -> Optional[np.ndarray]:
    """Extract face embedding from an image"""
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_image)
        
        if not results.detections:
            return None
        
        embedding = crop_face_and_embed(image, results.detections[0])
        return embedding

def save_face_image(student_id: int, image: np.ndarray, image_index: int) -> str:
    """Save face image to dataset folder"""
    student_folder = os.path.join(DATASET_DIR, str(student_id))
    os.makedirs(student_folder, exist_ok=True)
    
    filename = f"face_{image_index}.jpg"
    filepath = os.path.join(student_folder, filename)
    cv2.imwrite(filepath, image)
    
    return filepath

def train_model(progress_callback: Optional[Callable] = None) -> bool:
    """Train face recognition model"""
    if progress_callback:
        progress_callback(0, "Starting training...")
    
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        X = []
        y = []
        
        student_dirs = [d for d in os.listdir(DATASET_DIR) 
                       if os.path.isdir(os.path.join(DATASET_DIR, d))]
        
        if not student_dirs:
            if progress_callback:
                progress_callback(0, "No training data found")
            return False
        
        total_students = len(student_dirs)
        
        for idx, student_id in enumerate(student_dirs):
            folder_path = os.path.join(DATASET_DIR, student_id)
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in image_files:
                img_path = os.path.join(folder_path, img_file)
                image = cv2.imread(img_path)
                
                if image is None:
                    continue
                
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_image)
                
                if not results.detections:
                    continue
                
                embedding = crop_face_and_embed(image, results.detections[0])
                if embedding is None:
                    continue
                
                X.append(embedding)
                y.append(int(student_id))
            
            if progress_callback:
                progress = int((idx + 1) / total_students * 80)
                progress_callback(progress, f"Processing student {idx + 1}/{total_students}")
        
        if len(X) == 0:
            if progress_callback:
                progress_callback(0, "No valid training data found")
            return False
        
        if progress_callback:
            progress_callback(85, "Training model...")
        
        X = np.array(X)
        y = np.array(y)
        
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        classifier.fit(X, y)
        
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(classifier, f)
        
        if progress_callback:
            progress_callback(100, "Training complete!")
        
        return True

def load_model() -> Optional[RandomForestClassifier]:
    """Load trained model"""
    if not os.path.exists(MODEL_PATH):
        return None
    
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def predict_face(image: np.ndarray, confidence_threshold: float = 0.6) -> Tuple[Optional[int], float]:
    """Predict student ID from face image"""
    model = load_model()
    if model is None:
        return None, 0.0
    
    embedding = extract_face_embedding(image)
    if embedding is None:
        return None, 0.0
    
    probabilities = model.predict_proba([embedding])[0]
    max_idx = np.argmax(probabilities)
    confidence = float(probabilities[max_idx])
    
    if confidence < confidence_threshold:
        return None, confidence
    
    student_id = model.classes_[max_idx]
    return int(student_id), confidence

def is_model_trained() -> bool:
    """Check if model is trained"""
    return os.path.exists(MODEL_PATH)

def delete_student_images(student_id: int):
    """Delete all images for a student"""
    student_folder = os.path.join(DATASET_DIR, str(student_id))
    if os.path.exists(student_folder):
        import shutil
        shutil.rmtree(student_folder, ignore_errors=True)
