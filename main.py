import mediapipe as mp
import cv2
import pickle
import time
import os
import numpy as np

class ModelLoader:
    def __init__(self, models_base_path="model_baru"):
        self.models_base_path = models_base_path
        self.right_hand_model = None
        self.right_hand_scaler = None
        self.left_hand_model = None
        self.left_hand_scaler = None
        self.right_algorithm = None
        self.left_algorithm = None
        
    def find_best_model(self, hand_type):
        """Find the best model for specified hand"""
        best_info_path = os.path.join(self.models_base_path, f"BEST_{hand_type}", "best_model_info.txt")
        
        if os.path.exists(best_info_path):
            with open(best_info_path, 'r') as f:
                content = f.read()
                # Extract algorithm name
                for line in content.split('\n'):
                    if line.startswith('Best Algorithm:'):
                        algorithm = line.split(':')[1].strip()
                        return algorithm
        
        # Fallback: look for any available model
        if os.path.exists(self.models_base_path):
            for item in os.listdir(self.models_base_path):
                if hand_type in item and not item.startswith('BEST_'):
                    # Extract algorithm from folder name (e.g., "RF_kiri" -> "RF")
                    algorithm = item.split('_')[0]
                    return algorithm
        
        return None
    
    def load_model(self, hand_type):
        """Load model for specified hand"""
        algorithm = self.find_best_model(hand_type)
        if not algorithm:
            print(f"❌ No model found for {hand_type} hand")
            return None, None, None
        
        model_dir = os.path.join(self.models_base_path, f"{algorithm}_{hand_type}")
        
        if not os.path.exists(model_dir):
            print(f"❌ Model directory not found: {model_dir}")
            return None, None, None
        
        try:
            # Load model
            model_path = os.path.join(model_dir, "model.pkl")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load scaler (if exists)
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            scaler = None
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            
            print(f"✅ {hand_type} hand model loaded: {algorithm}")
            return model, scaler, algorithm
            
        except Exception as e:
            print(f"❌ Error loading {hand_type} model: {e}")
            return None, None, None
    
    def load_all_models(self):
        """Load both hand models"""
        print("🤖 Loading gesture recognition models...")
        
        # Load right hand model (notes)
        self.right_hand_model, self.right_hand_scaler, self.right_algorithm = self.load_model("kanan")
        
        # Load left hand model (height)
        self.left_hand_model, self.left_hand_scaler, self.left_algorithm = self.load_model("kiri")
        
        print(f"📊 Right hand: {self.right_algorithm if self.right_algorithm else 'Not available'}")
        print(f"📊 Left hand: {self.left_algorithm if self.left_algorithm else 'Not available'}")
        
        return (self.right_hand_model is not None) or (self.left_hand_model is not None)

def load_class_names():
    with open(r"gestures.names", 'r') as f:
        class_names = f.read().split('\n')
    print(f"📋 Classes loaded: {class_names}")
    return class_names

def extract_hand_landmarks(hand_landmarks):
    """Extract 3D landmarks from hand"""
    features = []
    for landmark in hand_landmarks.landmark:
        features.extend([landmark.x, landmark.y, landmark.z])
    return np.array(features).reshape(1, -1)

def predict_gesture(model, scaler, features):
    """Make prediction using loaded model"""
    try:
        if scaler is not None:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features
        
        prediction = model.predict(features_scaled)
        
        # Get prediction probability if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)
            confidence = np.max(probabilities)
        else:
            confidence = 1.0
        
        return prediction[0], confidence
        
    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        return None, 0.0

def run_smart_dual_hand_demo(cap, class_names, count_fps=False):
    """Run demo with new ML models"""
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Load models using new model loader
    model_loader = ModelLoader()
    if not model_loader.load_all_models():
        print("❌ No models available! Using fallback...")
        return

    # Main loop variables
    confidence_threshold = 0.7  # Minimum confidence for predictions

    with mp_hands.Hands(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5,
        max_num_hands=2  # Allow detection of both hands
    ) as hands:
        estimate_pose = True
        previous_frame_time = 0

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Camera Error")
                break
            
            # BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Flip horizontal (mirror)
            image = cv2.flip(image, 1)
            
            # MediaPipe detection
            results = hands.process(image)
            
            # RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            right_gesture = ""
            left_gesture = ""
            right_confidence = 0.0
            left_confidence = 0.0
            combined_result = ""
            
            # Process detected hands
            if results.multi_hand_landmarks and estimate_pose:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Draw landmarks
                    color = (250, 44, 250) if handedness.classification[0].label == "Right" else (44, 250, 44)
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
                    )

                    hand_type = handedness.classification[0].label
                    
                    # Extract features
                    features = extract_hand_landmarks(hand_landmarks)
                    
                    if hand_type == "Right" and model_loader.right_hand_model is not None:
                        # Right hand prediction (notes)
                        prediction, confidence = predict_gesture(
                            model_loader.right_hand_model,
                            model_loader.right_hand_scaler,
                            features
                        )
                        
                        if prediction and confidence >= confidence_threshold:
                            right_gesture = prediction
                            right_confidence = confidence
                            print(f"Right Hand (Note): {right_gesture.upper()} ({confidence:.2f})")
                    
                    elif hand_type == "Left" and model_loader.left_hand_model is not None:
                        # Left hand prediction (height)
                        prediction, confidence = predict_gesture(
                            model_loader.left_hand_model,
                            model_loader.left_hand_scaler,
                            features
                        )
                        
                        if prediction and confidence >= confidence_threshold:
                            left_gesture = prediction
                            left_confidence = confidence
                            print(f"Left Hand (Height): {left_gesture.upper()} ({confidence:.2f})")

            # Combine results
            if right_gesture and left_gesture:
                combined_result = f"{right_gesture.upper()} ({left_gesture.upper()})"
            elif right_gesture:
                combined_result = f"{right_gesture.upper()}"
            elif left_gesture:
                combined_result = f"Height: {left_gesture.upper()}"

            # FPS counter
            if count_fps:
                current_frame_time = time.time()
                fps = 1/(current_frame_time - previous_frame_time) if previous_frame_time > 0 else 0
                previous_frame_time = current_frame_time
                fps = int(fps)
                cv2.putText(image, f"FPS: {fps}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (128, 128, 128), 1, cv2.LINE_AA)

            # Display results
            y_offset = 55
            if estimate_pose:
                if combined_result:
                    cv2.putText(image, f"Gesture: {combined_result}", (5, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    y_offset += 30
                
                # Individual predictions with confidence
                if right_gesture:
                    cv2.putText(image, f"Right: {right_gesture.upper()} ({right_confidence:.2f})", (5, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (250, 44, 250), 1, cv2.LINE_AA)
                    y_offset += 25
                
                if left_gesture:
                    cv2.putText(image, f"Left: {left_gesture.upper()} ({left_confidence:.2f})", (5, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (44, 250, 44), 1, cv2.LINE_AA)
                    y_offset += 25
                
                # Model info
                right_model_info = f"Right: {model_loader.right_algorithm or 'N/A'}"
                left_model_info = f"Left: {model_loader.left_algorithm or 'N/A'}"
                cv2.putText(image, right_model_info, (5, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
                y_offset += 20
                cv2.putText(image, left_model_info, (5, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

            # Instructions
            cv2.putText(image, "Right hand: Pink | Left hand: Green", (5, image.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "SPACE: Toggle | ESC: Exit", (5, image.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Kodaly Smart Gesture Recognition", image)

            key = cv2.waitKey(1)
            if key == 32:  # SPACE
                estimate_pose = not estimate_pose
                print(f"📊 Gesture estimation: {'ON' if estimate_pose else 'OFF'}")
            if key == 27:  # ESCAPE
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("=== KODALY SMART GESTURE RECOGNITION ===")
    print("🎵 Dual SVM Models (Right: Notes | Left: Height)")
    print("Controls:")
    print("- Right hand (Pink): Note recognition (DO, RE, MI, FA, SOL, LA, SI)")
    print("- Left hand (Green): Height detection (LOW, MEDIUM, HIGH)")
    print("- SPACE: Toggle gesture estimation")
    print("- ESC: Exit")
    print()
    
    try:
        cap = cv2.VideoCapture(0)
        class_names = load_class_names()
        run_smart_dual_hand_demo(cap, class_names, count_fps=True)
    except Exception as e:
        print(f"Error: {e}")
