import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

class ModelTrainer:
    def __init__(self):
        self.dataset_path = "dataset_collection2"
        self.models_base_path = "model_baru"
        self.algorithms = {
            'RF': RandomForestClassifier,
            'KNN': KNeighborsClassifier, 
            'SVM': SVC
        }
        
    def load_csv_dataset(self, hand_type):
        """Load CSV dataset for specified hand (kiri/kanan)"""
        print(f"\n📁 Loading {hand_type} hand dataset...")
        
        hand_path = os.path.join(self.dataset_path, hand_type)
        if not os.path.exists(hand_path):
            print(f"❌ Path not found: {hand_path}")
            return None, None
        
        all_data = []
        classes = []
        
        # Get all class directories
        class_dirs = [d for d in os.listdir(hand_path) if os.path.isdir(os.path.join(hand_path, d))]
        print(f"📊 Found classes: {class_dirs}")
        
        for class_name in class_dirs:
            class_path = os.path.join(hand_path, class_name)
            csv_files = glob.glob(os.path.join(class_path, "*.csv"))
            
            print(f"  📋 Class '{class_name}': {len(csv_files)} files")
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    
                    # Extract features (exclude 'class' column)
                    if 'class' in df.columns:
                        features = df.drop('class', axis=1).iloc[0].values
                        class_label = df['class'].iloc[0]
                    else:
                        # If no class column, use filename
                        features = df.iloc[0].values
                        class_label = class_name
                    
                    all_data.append(features)
                    classes.append(class_label)
                    
                except Exception as e:
                    print(f"    ⚠️ Error loading {csv_file}: {e}")
        
        if not all_data:
            print(f"❌ No data loaded for {hand_type}")
            return None, None
        
        # Convert to numpy arrays
        X = np.array(all_data)
        y = np.array(classes)
        
        # Print dataset info
        print(f"✅ Dataset loaded:")
        print(f"   📊 Total samples: {len(X)}")
        print(f"   🎯 Features: {X.shape[1]}")
        print(f"   📈 Classes: {np.unique(y)}")
        print(f"   📊 Class distribution:")
        class_counts = Counter(y)
        for class_name, count in class_counts.items():
            print(f"      {class_name}: {count}")
        
        return X, y
    
    def train_algorithm(self, X_train, X_test, y_train, y_test, algorithm_name, hand_type):
        """Train and evaluate a specific algorithm"""
        print(f"\n🤖 Training {algorithm_name} for {hand_type} hand...")
        
        # Define parameter grids for each algorithm
        param_grids = {
            'RF': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            }
        }
        
        # Scale data for KNN and SVM
        if algorithm_name in ['KNN', 'SVM']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
            scaler = None
        
        # Initialize model
        model_class = self.algorithms[algorithm_name]
        if algorithm_name == 'SVM':
            base_model = model_class(random_state=42, probability=True)
        elif algorithm_name == 'RF':
            base_model = model_class(random_state=42)
        else:  # KNN doesn't have random_state
            base_model = model_class()
        
        # Grid search for best parameters
        print(f"   🔍 Searching best parameters...")
        grid_search = GridSearchCV(
            base_model, 
            param_grids[algorithm_name],
            cv=3,  # 3-fold cross validation
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"   ✅ Best parameters: {best_params}")
        
        # Predictions
        y_pred = best_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"   📊 Test Accuracy: {accuracy:.4f}")
        print(f"   📊 CV Score: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
        
        # Detailed report
        print(f"   📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=np.unique(y_test)))
        
        return {
            'model': best_model,
            'scaler': scaler,
            'accuracy': accuracy,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'params': best_params,
            'algorithm': algorithm_name
        }
    
    def save_model(self, model_info, hand_type):
        """Save model and related files"""
        algorithm = model_info['algorithm']
        accuracy = model_info['accuracy']
        
        # Create directory
        model_dir = os.path.join(self.models_base_path, f"{algorithm}_{hand_type}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_info['model'], f)
        
        # Save scaler if exists
        if model_info['scaler'] is not None:
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(model_info['scaler'], f)
        
        # Save model info
        info_path = os.path.join(model_dir, "model_info.pkl")
        with open(info_path, 'wb') as f:
            pickle.dump(model_info, f)
        
        print(f"   💾 Model saved to: {model_dir}")
        print(f"   📊 Accuracy: {accuracy:.4f}")
        
        return model_dir
    
    def train_hand_models(self, hand_type):
        """Train all algorithms for specified hand"""
        print(f"\n{'='*50}")
        print(f"🎯 TRAINING {hand_type.upper()} HAND MODELS")
        print(f"{'='*50}")
        
        # Load dataset
        X, y = self.load_csv_dataset(hand_type)
        if X is None:
            print(f"❌ Failed to load dataset for {hand_type}")
            return None
        
        # Check minimum samples per class
        class_counts = Counter(y)
        min_samples = min(class_counts.values())
        if min_samples < 2:
            print(f"❌ Insufficient samples. Minimum class has only {min_samples} samples.")
            return None
        
        # Split dataset
        test_size = 0.2 if len(X) > 10 else 0.3  # Adjust test size for small datasets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=42, 
            stratify=y
        )
        
        print(f"📊 Train samples: {len(X_train)}")
        print(f"📊 Test samples: {len(X_test)}")
        
        # Train all algorithms
        results = {}
        
        for algorithm_name in self.algorithms.keys():
            try:
                result = self.train_algorithm(X_train, X_test, y_train, y_test, algorithm_name, hand_type)
                results[algorithm_name] = result
                
                # Save model
                self.save_model(result, hand_type)
                
            except Exception as e:
                print(f"❌ Error training {algorithm_name}: {e}")
        
        # Find best algorithm
        if results:
            best_algorithm = max(results.keys(), key=lambda k: results[k]['accuracy'])
            best_result = results[best_algorithm]
            
            print(f"\n🏆 BEST ALGORITHM FOR {hand_type.upper()} HAND:")
            print(f"   🥇 Algorithm: {best_algorithm}")
            print(f"   📊 Accuracy: {best_result['accuracy']:.4f}")
            print(f"   📊 CV Score: {best_result['cv_mean']:.4f} (+/- {best_result['cv_std'] * 2:.4f})")
            
            # Create symlink or copy for best model
            best_model_dir = os.path.join(self.models_base_path, f"BEST_{hand_type}")
            os.makedirs(best_model_dir, exist_ok=True)
            
            # Save best model info
            best_info_path = os.path.join(best_model_dir, "best_model_info.txt")
            with open(best_info_path, 'w') as f:
                f.write(f"Best Algorithm: {best_algorithm}\n")
                f.write(f"Accuracy: {best_result['accuracy']:.4f}\n")
                f.write(f"CV Score: {best_result['cv_mean']:.4f} (+/- {best_result['cv_std'] * 2:.4f})\n")
                f.write(f"Parameters: {best_result['params']}\n")
                f.write(f"Model Path: {self.models_base_path}/{best_algorithm}_{hand_type}/\n")
            
            return results, best_algorithm
        
        return None, None
    
    def train_all_models(self):
        """Train models for both hands"""
        print("🎵 KODALY GESTURE RECOGNITION - MODEL TRAINING")
        print("="*60)
        
        # Create base directory
        os.makedirs(self.models_base_path, exist_ok=True)
        
        results = {}
        
        # Train left hand models (height detection)
        left_results, best_left = self.train_hand_models("kiri")
        if left_results:
            results['kiri'] = {'results': left_results, 'best': best_left}
        
        # Train right hand models (note detection) 
        right_results, best_right = self.train_hand_models("kanan")
        if right_results:
            results['kanan'] = {'results': right_results, 'best': best_right}
        
        # Summary
        print(f"\n{'='*60}")
        print("🎉 TRAINING SUMMARY")
        print(f"{'='*60}")
        
        for hand_type, hand_results in results.items():
            print(f"\n{hand_type.upper()} HAND:")
            if hand_results['results']:
                for alg, result in hand_results['results'].items():
                    status = "🥇" if alg == hand_results['best'] else "  "
                    print(f"  {status} {alg}: {result['accuracy']:.4f}")
                print(f"  🏆 Best: {hand_results['best']}")
            else:
                print("  ❌ No models trained")
        
        return results

if __name__ == "__main__":
    trainer = ModelTrainer()
    results = trainer.train_all_models()
    
    print(f"\n✅ Training completed!")
    print(f"📁 Models saved in: {trainer.models_base_path}/")
    print(f"📋 Available models:")
    
    # List created models
    if os.path.exists(trainer.models_base_path):
        for item in os.listdir(trainer.models_base_path):
            item_path = os.path.join(trainer.models_base_path, item)
            if os.path.isdir(item_path):
                print(f"   📂 {item}")
