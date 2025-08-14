import mediapipe as mp
import cv2
import os
import time
import csv
from datetime import datetime

class DatasetCollector:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        
        # Dataset configuration
        self.base_dataset_dir = "dataset"
        
        # Class definitions
        self.left_hand_classes = ["low", "medium", "high"]
        self.right_hand_classes = ["do", "re", "mi", "fa", "sol", "la", "si"]
        
        # Current collection settings
        self.current_hand = None 
        self.current_class = None
        self.is_collecting = False
        self.collected_samples = 0
        self.samples_per_class = 100
        
        # Data storage
        self.current_data = []
        
        self.setup_directories()
    
    def setup_directories(self):
        """Create directory structure for dataset collection"""
        if not os.path.exists(self.base_dataset_dir):
            os.makedirs(self.base_dataset_dir)
        
        # Create subdirectories for left hand (height)
        left_dir = os.path.join(self.base_dataset_dir, "kiri")
        if not os.path.exists(left_dir):
            os.makedirs(left_dir)
        for class_name in self.left_hand_classes:
            class_dir = os.path.join(left_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
        
        # Create subdirectories for right hand (notes)
        right_dir = os.path.join(self.base_dataset_dir, "kanan")
        if not os.path.exists(right_dir):
            os.makedirs(right_dir)
        for class_name in self.right_hand_classes:
            class_dir = os.path.join(right_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
        
        print("✅ Directory structure created:")
        print(f"  📁 {self.base_dataset_dir}/")
        print(f"    📁 kiri/ (left hand - height)")
        for class_name in self.left_hand_classes:
            print(f"      📁 {class_name}/")
        print(f"    📁 kanan/ (right hand - notes)")
        for class_name in self.right_hand_classes:
            print(f"      📁 {class_name}/")
    
    def select_hand_and_class(self):
        """Interactive selection of hand and class for data collection"""
        print("\n" + "="*50)
        print("📊 DATASET COLLECTION SETUP")
        print("="*50)
        
        # Select hand
        while True:
            print("\n🤲 Pilih tangan untuk pengumpulan dataset:")
            print("1. Kiri (Height: low, medium, high)")
            print("2. Kanan (Nada: do, re, mi, fa, sol, la, si)")
            print("0. Keluar")
            
            choice = input("\nPilihan (0-2): ").strip()
            
            if choice == "0":
                return False
            elif choice == "1":
                self.current_hand = "kiri"
                available_classes = self.left_hand_classes
                break
            elif choice == "2":
                self.current_hand = "kanan"
                available_classes = self.right_hand_classes
                break
            else:
                print("❌ Pilihan tidak valid!")
        
        # Select class
        while True:
            print(f"\n🎯 Pilih kelas untuk tangan {self.current_hand}:")
            for i, class_name in enumerate(available_classes, 1):
                print(f"{i}. {class_name.upper()}")
            print("0. Kembali")
            
            choice = input(f"\nPilihan (0-{len(available_classes)}): ").strip()
            
            if choice == "0":
                return self.select_hand_and_class()
            elif choice.isdigit() and 1 <= int(choice) <= len(available_classes):
                self.current_class = available_classes[int(choice) - 1]
                break
            else:
                print("❌ Pilihan tidak valid!")
        
        print(f"\n✅ Dipilih: Tangan {self.current_hand.upper()} - Kelas {self.current_class.upper()}")
        print(f"📊 Target: {self.samples_per_class} samples")
        
        return True
    
    def extract_right_hand_features(self, hand_landmarks):
        """Extract features for right hand (3D landmarks)"""
        features = []
        for landmark in hand_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        return features
    
    def extract_left_hand_features(self, hand_landmarks):
        """Extract features for left hand (hand landmarks only)"""
        features = []
        
        # Add all hand landmarks (21 landmarks × 3 coordinates = 63 features)
        for landmark in hand_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        
        return features
    
    def save_sample(self, features):
        """Save a single sample to the current dataset"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Create filename
        filename = f"{self.current_class}_{timestamp}_{self.collected_samples:04d}.csv"
        filepath = os.path.join(self.base_dataset_dir, self.current_hand, self.current_class, filename)
        
        # Save features to CSV
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            header = [f"landmark_{i}_{axis}" for i in range(21) for axis in ['x', 'y', 'z']] + ['class']
            writer.writerow(header)
            
            # Write data
            row = features + [self.current_class]
            writer.writerow(row)
        
        self.collected_samples += 1
        print(f"💾 Saved sample {self.collected_samples}/{self.samples_per_class} - {filename}")
    
    def collect_data(self):
        """Main data collection loop"""
        cap = cv2.VideoCapture(0)
        
        # Initialize MediaPipe
        hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1
        )
        
        print(f"\n🎥 Mulai pengumpulan data untuk: {self.current_hand.upper()} - {self.current_class.upper()}")
        print("📋 Kontrol:")
        print("  SPACE: Mulai/Berhenti pengumpulan")
        print("  R: Reset counter")
        print("  ESC: Selesai dan simpan")
        print("  Q: Keluar tanpa menyimpan")
        
        last_save_time = 0
        save_interval = 0.1  # Save every 100ms when collecting
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Tidak dapat membaca kamera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hands
            hand_results = hands.process(rgb_frame)
            
            # Draw landmarks and collect data
            target_hand_detected = False
            
            if hand_results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                    hand_label = handedness.classification[0].label
                    
                    # Check if this is the target hand
                    is_target_hand = (
                        (self.current_hand == "kanan" and hand_label == "Right") or
                        (self.current_hand == "kiri" and hand_label == "Left")
                    )
                    
                    if is_target_hand:
                        target_hand_detected = True
                        
                        # Draw hand landmarks
                        color = (0, 255, 0) if is_target_hand else (128, 128, 128)
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
                        )
                        
                        # Collect data if active
                        if self.is_collecting and time.time() - last_save_time > save_interval:
                            try:
                                if self.current_hand == "kanan":
                                    features = self.extract_right_hand_features(hand_landmarks)
                                    if len(features) == 63:  # 21 landmarks × 3 coordinates
                                        self.save_sample(features)
                                        last_save_time = time.time()
                                else:  # left hand
                                    features = self.extract_left_hand_features(hand_landmarks)
                                    if len(features) == 63:  # 21 landmarks × 3 coordinates
                                        self.save_sample(features)
                                        last_save_time = time.time()
                                    else:
                                        print(f"⚠️ Feature count mismatch: {len(features)} (expected 63)")
                            except Exception as e:
                                print(f"❌ Error saving sample: {e}")
                        
                        # Stop collecting if target reached
                        if self.collected_samples >= self.samples_per_class:
                            self.is_collecting = False
                            print(f"🎉 Target tercapai! {self.samples_per_class} samples terkumpul.")
            
            # Display information
            info_y = 30
            cv2.putText(frame, f"Hand: {self.current_hand.upper()}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            info_y += 30
            cv2.putText(frame, f"Class: {self.current_class.upper()}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            info_y += 30
            cv2.putText(frame, f"Samples: {self.collected_samples}/{self.samples_per_class}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            info_y += 30
            
            # Collection status
            status_color = (0, 255, 0) if self.is_collecting else (0, 0, 255)
            status_text = "COLLECTING" if self.is_collecting else "PAUSED"
            cv2.putText(frame, f"Status: {status_text}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            info_y += 30
            
            # Hand detection status
            detection_color = (0, 255, 0) if target_hand_detected else (0, 0, 255)
            detection_text = "DETECTED" if target_hand_detected else "NOT DETECTED"
            cv2.putText(frame, f"Target Hand: {detection_text}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, detection_color, 2)
            
            # Controls
            cv2.putText(frame, "SPACE: Start/Stop | R: Reset | ESC: Finish | Q: Quit", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Dataset Collection", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space - toggle collection
                if target_hand_detected:
                    self.is_collecting = not self.is_collecting
                    status = "dimulai" if self.is_collecting else "dihentikan"
                    print(f"📊 Pengumpulan data {status}")
                else:
                    print("⚠️ Tangan target belum terdeteksi!")
            elif key == ord('r') or key == ord('R'):  # Reset
                self.collected_samples = 0
                print("🔄 Counter direset")
            elif key == 27:  # ESC - finish
                print(f"✅ Pengumpulan selesai. Total samples: {self.collected_samples}")
                break
            elif key == ord('q') or key == ord('Q'):  # Quit
                print("❌ Keluar tanpa menyimpan")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
    
    def run(self):
        """Main program loop"""
        print("🎵 KODALY DATASET COLLECTOR")
        print("="*50)
        
        while True:
            if not self.select_hand_and_class():
                break
            
            # Reset collection state
            self.is_collecting = False
            self.collected_samples = 0
            
            # Start data collection
            self.collect_data()
            
            # Ask if user wants to continue
            print("\n" + "="*50)
            while True:
                continue_choice = input("🔄 Lanjut mengumpulkan dataset lain? (y/n): ").strip().lower()
                if continue_choice in ['y', 'yes', 'ya']:
                    break
                elif continue_choice in ['n', 'no', 'tidak']:
                    print("👋 Terima kasih! Dataset collection selesai.")
                    return
                else:
                    print("❌ Pilihan tidak valid! Masukkan 'y' atau 'n'")

if __name__ == "__main__":
    try:
        collector = DatasetCollector()
        collector.run()
    except KeyboardInterrupt:
        print("\n\n⚠️ Program dihentikan oleh user")
    except Exception as e:
        print(f"❌ Error: {e}")
