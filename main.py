"""
SPEECH SCORING COMPETITION SOLUTION
Complete pipeline with training and test results
Author: Competition Helper
Date: 2024
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Check and install required packages
def install_requirements():
    """Install required packages if missing"""
    required_packages = [
        'pandas',
        'numpy', 
        'scikit-learn',
        'joblib',
        'librosa',
        'tqdm',
        'soundfile',
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"📦 Installing missing packages: {missing_packages}")
        import subprocess
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
                print(f"  ✅ Installed: {package}")
            except:
                print(f"  ❌ Failed to install: {package}")
        print("✅ All packages installed!\n")
    else:
        print("✅ All required packages are already installed!\n")

# Install requirements first
install_requirements()

# Now import everything
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class SpeechScoringCompetition:
    """
    Complete solution for speech scoring competition.
    Handles your specific folder structure: dataset/audios/ and dataset/csvs/
    """
    
    def __init__(self, data_path="dataset"):
        self.data_path = data_path
        self.train_audio_dir = os.path.join(data_path, "audios", "train")
        self.test_audio_dir = os.path.join(data_path, "audios", "test")
        self.train_csv = os.path.join(data_path, "csvs", "train.csv")
        self.test_csv = os.path.join(data_path, "csvs", "test.csv")
        
        # Initialize results storage
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.train_filenames = []
        self.test_filenames = []
        self.model = None
        self.scaler = None
        self.train_predictions = None
        self.test_predictions = None
        self.metrics = {}
        
    def validate_structure(self):
        """Validate the folder and file structure"""
        print("🔍 Validating data structure...")
        print("-" * 50)
        
        # Check main directories
        required_dirs = [
            ("dataset/", self.data_path),
            ("dataset/audios/", os.path.join(self.data_path, "audios")),
            ("dataset/audios/train/", self.train_audio_dir),
            ("dataset/audios/test/", self.test_audio_dir),
            ("dataset/csvs/", os.path.join(self.data_path, "csvs"))
        ]
        
        for dir_name, dir_path in required_dirs:
            if os.path.exists(dir_path):
                print(f"✅ {dir_name}")
            else:
                print(f"❌ {dir_name} - NOT FOUND")
                return False
        
        # Check CSV files
        required_files = [
            ("dataset/csvs/train.csv", self.train_csv),
            ("dataset/csvs/test.csv", self.test_csv)
        ]
        
        for file_name, file_path in required_files:
            if os.path.exists(file_path):
                print(f"✅ {file_name}")
            else:
                print(f"❌ {file_name} - NOT FOUND")
                return False
        
        # Check for audio files
        try:
            train_files = os.listdir(self.train_audio_dir)
            test_files = os.listdir(self.test_audio_dir)
            
            # Count .wav files
            train_wav = [f for f in train_files if f.lower().endswith('.wav')]
            test_wav = [f for f in test_files if f.lower().endswith('.wav')]
            
            print(f"\n🎵 Audio files found:")
            print(f"   Training: {len(train_wav)} .wav files")
            print(f"   Test: {len(test_wav)} .wav files")
            
            if len(train_wav) == 0:
                print("⚠️  Warning: No .wav files in training folder")
            if len(test_wav) == 0:
                print("⚠️  Warning: No .wav files in test folder")
                
        except Exception as e:
            print(f"❌ Error reading audio directories: {e}")
            return False
        
        print("-" * 50)
        print("✅ Data structure validation passed!")
        return True
    
    def load_csv_data(self):
        """Load and validate CSV files"""
        print("\n📊 Loading CSV data...")
        
        try:
            # Load training CSV
            self.train_df = pd.read_csv(self.train_csv)
            print(f"✅ Training CSV loaded: {len(self.train_df)} rows")
            print(f"   Columns: {list(self.train_df.columns)}")
            
            # Check required columns
            if 'filename' not in self.train_df.columns:
                print("❌ 'filename' column not found in train.csv")
                return False
            if 'label' not in self.train_df.columns:
                print("❌ 'label' column not found in train.csv")
                return False
            
            # Show label statistics
            print(f"   Label stats - Min: {self.train_df['label'].min():.2f}, "
                  f"Max: {self.train_df['label'].max():.2f}, "
                  f"Mean: {self.train_df['label'].mean():.2f}")
            
            # Load test CSV
            self.test_df = pd.read_csv(self.test_csv)
            print(f"✅ Test CSV loaded: {len(self.test_df)} rows")
            print(f"   Columns: {list(self.test_df.columns)}")
            
            if 'filename' not in self.test_df.columns:
                print("❌ 'filename' column not found in test.csv")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading CSV files: {e}")
            return False
    
    def find_audio_file(self, filename, directory):
        """Find audio file with various extensions"""
        # Try different variations
        basename = os.path.splitext(filename)[0]
        possible_names = [
            filename,
            filename + '.wav',
            filename + '.WAV',
            basename + '.wav',
            basename + '.WAV',
            filename.replace('.mp3', '.wav'),
            filename.replace('.m4a', '.wav')
        ]
        
        for name in possible_names:
            file_path = os.path.join(directory, name)
            if os.path.exists(file_path):
                return file_path
        
        return None
    
    def extract_audio_features(self, audio_path, sr=16000, max_duration=5):
        """
        Extract 25 features from audio file
        Returns: List of 25 feature values
        """
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=sr, duration=max_duration)
            
            # If audio is empty, return zeros
            if len(audio) == 0:
                return [0.0] * 25
            
            features = []
            
            # 1. Basic Statistics (5 features)
            features.append(len(audio) / sr)  # Duration
            features.append(np.max(np.abs(audio)))  # Max amplitude
            features.append(np.mean(np.abs(audio)))  # Mean amplitude
            features.append(np.std(audio))  # Standard deviation
            features.append(np.median(np.abs(audio)))  # Median amplitude
            
            # 2. Energy Features (3 features)
            features.append(np.mean(audio ** 2))  # RMS energy
            features.append(librosa.feature.rms(y=audio).mean())  # RMS
            features.append(np.sum(audio ** 2) / len(audio))  # Energy
            
            # 3. Zero Crossing Rate (2 features)
            zcr = librosa.feature.zero_crossing_rate(audio)
            features.append(zcr.mean())
            features.append(zcr.std())
            
            # 4. Spectral Features (5 features)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features.append(spectral_centroid.mean())
            features.append(spectral_centroid.std())
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            features.append(spectral_bandwidth.mean())
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            features.append(spectral_rolloff.mean())
            
            # 5. MFCC Features (5 features - mean of first 5 coefficients)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            for i in range(5):
                features.append(mfcc[i].mean())
            
            # 6. Chroma Features (2 features)
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features.append(chroma.mean())
            features.append(chroma.std())
            
            # 7. Pitch Features (2 features)
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitches = pitches[pitches > 0]
            if len(pitches) > 0:
                features.append(np.mean(pitches))
                features.append(np.std(pitches))
            else:
                features.extend([0.0, 0.0])
            
            # Ensure we have exactly 25 features
            if len(features) < 25:
                features.extend([0.0] * (25 - len(features)))
            
            return features[:25]
            
        except Exception as e:
            print(f"⚠️  Error processing {os.path.basename(audio_path)}: {str(e)[:50]}...")
            return [0.0] * 25
    
    def process_dataset(self, df, audio_dir, is_training=True):
        """
        Process all files in a dataset
        Returns: features array, labels array (if training), valid indices
        """
        features_list = []
        labels_list = []
        valid_indices = []
        filenames_list = []
        
        desc = "Training" if is_training else "Test"
        
        print(f"\n🎵 Processing {desc} audio files...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{desc} files"):
            filename = row['filename']
            
            # Find the audio file
            audio_path = self.find_audio_file(filename, audio_dir)
            
            if audio_path is None:
                print(f"⚠️  Could not find audio file: {filename}")
                continue
            
            # Extract features
            features = self.extract_audio_features(audio_path)
            features_list.append(features)
            valid_indices.append(idx)
            filenames_list.append(filename)
            
            # Get label if training
            if is_training:
                labels_list.append(row['label'])
        
        # Convert to numpy arrays
        features_array = np.array(features_list)
        
        if is_training:
            return features_array, np.array(labels_list), valid_indices, filenames_list
        else:
            return features_array, valid_indices, filenames_list
    
    def train_model(self, X_train, y_train):
        """Train the model with cross-validation"""
        print("\n🤖 Training model...")
        print("-" * 40)
        
        # Split for validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, shuffle=True
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_split)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Initialize model
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        # Perform cross-validation
        print("   Performing 5-fold cross-validation...")
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train_split, 
            cv=5, scoring='r2', n_jobs=-1
        )
        
        print(f"   Cross-validation R² scores: {cv_scores}")
        print(f"   Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train the model
        print("   Training final model...")
        self.model.fit(X_train_scaled, y_train_split)
        
        # Validate on validation set
        y_val_pred = self.model.predict(X_val_scaled)
        
        # Calculate validation metrics
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        print(f"\n📊 Validation Set Performance:")
        print(f"   RMSE: {val_rmse:.4f}")
        print(f"   MAE: {val_mae:.4f}")
        print(f"   R²: {val_r2:.4f}")
        
        # Store metrics
        self.metrics['cv_scores'] = cv_scores
        self.metrics['val_rmse'] = val_rmse
        self.metrics['val_mae'] = val_mae
        self.metrics['val_r2'] = val_r2
        
        # Retrain on full training set
        print("\n🤖 Retraining on full training data...")
        X_full_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_full_scaled, y_train)
        
        # Predict on training set for analysis
        self.train_predictions = self.model.predict(X_full_scaled)
        
        # Calculate training metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, self.train_predictions))
        train_mae = mean_absolute_error(y_train, self.train_predictions)
        train_r2 = r2_score(y_train, self.train_predictions)
        
        print(f"\n📊 Full Training Set Performance:")
        print(f"   RMSE: {train_rmse:.4f}")
        print(f"   MAE: {train_mae:.4f}")
        print(f"   R²: {train_r2:.4f}")
        
        self.metrics['train_rmse'] = train_rmse
        self.metrics['train_mae'] = train_mae
        self.metrics['train_r2'] = train_r2
        
        return self.model
    
    def predict_test(self, X_test):
        """Make predictions on test data"""
        print("\n🔮 Making predictions on test data...")
        
        if self.model is None or self.scaler is None:
            print("❌ Model not trained yet!")
            return None
        
        # Scale test features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        self.test_predictions = self.model.predict(X_test_scaled)
        
        # Clip predictions to reasonable range (0-100)
        self.test_predictions = np.clip(self.test_predictions, 0, 100)
        
        print(f"✅ Predictions made for {len(self.test_predictions)} test files")
        print(f"   Min score: {self.test_predictions.min():.2f}")
        print(f"   Max score: {self.test_predictions.max():.2f}")
        print(f"   Mean score: {self.test_predictions.mean():.2f}")
        
        return self.test_predictions
    
    def create_visualizations(self, y_train_true, y_train_pred):
        """Create visualization plots"""
        print("\n📈 Creating visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. True vs Predicted Scatter
        axes[0, 0].scatter(y_train_true, y_train_pred, alpha=0.6, edgecolors='w', linewidth=0.5)
        axes[0, 0].plot([y_train_true.min(), y_train_true.max()], 
                       [y_train_true.min(), y_train_true.max()], 
                       'r--', lw=2, label='Perfect prediction')
        axes[0, 0].set_xlabel('True Scores')
        axes[0, 0].set_ylabel('Predicted Scores')
        axes[0, 0].set_title('True vs Predicted Scores')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residual Plot
        residuals = y_train_pred - y_train_true
        axes[0, 1].scatter(y_train_pred, residuals, alpha=0.6, edgecolors='w', linewidth=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Scores')
        axes[0, 1].set_ylabel('Residuals (Predicted - True)')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Error Distribution
        axes[0, 2].hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[0, 2].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[0, 2].set_xlabel('Prediction Error')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Error Distribution')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Distribution Comparison
        axes[1, 0].hist(y_train_true, bins=30, alpha=0.5, label='True Scores', density=True)
        axes[1, 0].hist(y_train_pred, bins=30, alpha=0.5, label='Predicted Scores', density=True)
        axes[1, 0].set_xlabel('Scores')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Score Distributions')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Prediction Error vs True Score
        axes[1, 1].scatter(y_train_true, np.abs(residuals), alpha=0.6, 
                          edgecolors='w', linewidth=0.5, color='purple')
        axes[1, 1].set_xlabel('True Scores')
        axes[1, 1].set_ylabel('Absolute Error')
        axes[1, 1].set_title('Absolute Error vs True Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Box plot comparison
        axes[1, 2].boxplot([y_train_true, y_train_pred], 
                          labels=['True Scores', 'Predicted Scores'])
        axes[1, 2].set_ylabel('Score Values')
        axes[1, 2].set_title('Score Distribution Comparison')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_performance.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Create test predictions distribution plot
        if self.test_predictions is not None:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.hist(self.test_predictions, bins=30, alpha=0.7, color='orange', edgecolor='black')
            ax2.set_xlabel('Predicted Scores')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Test Set Predictions Distribution')
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('test_predictions_distribution.png', dpi=150, bbox_inches='tight')
            plt.show()
    
    def create_submission_file(self):
        """Create competition submission file"""
        print("\n💾 Creating submission file...")
        
        # Create submission dataframe
        submission_df = self.test_df.copy()
        
        # Initialize predictions column
        submission_df['predicted_score'] = np.nan
        
        # Fill predictions for files we processed
        if hasattr(self, 'test_indices') and hasattr(self, 'test_predictions'):
            for idx, pred in zip(self.test_indices, self.test_predictions):
                if idx < len(submission_df):
                    submission_df.loc[idx, 'predicted_score'] = pred
        
        # Fill any NaN values with median
        median_score = np.nanmedian(submission_df['predicted_score'])
        if np.isnan(median_score):
            median_score = 70.0  # Default value
        
        submission_df['predicted_score'] = submission_df['predicted_score'].fillna(median_score)
        
        # Clip to 0-100 range
        submission_df['predicted_score'] = submission_df['predicted_score'].clip(0, 100)
        
        # Save to CSV
        submission_file = 'submission.csv'
        submission_df[['filename', 'predicted_score']].to_csv(submission_file, index=False)
        
        print(f"✅ Submission file created: {submission_file}")
        print(f"\n📊 Submission Statistics:")
        print(f"   Total predictions: {len(submission_df)}")
        print(f"   Mean score: {submission_df['predicted_score'].mean():.2f}")
        print(f"   Std score: {submission_df['predicted_score'].std():.2f}")
        print(f"   Min score: {submission_df['predicted_score'].min():.2f}")
        print(f"   Max score: {submission_df['predicted_score'].max():.2f}")
        
        return submission_file, submission_df
    
    def save_results(self):
        """Save all results and models"""
        print("\n💾 Saving all results...")
        
        # 1. Save model and scaler
        joblib.dump(self.model, 'speech_scoring_model.pkl')
        joblib.dump(self.scaler, 'feature_scaler.pkl')
        print("✅ Model saved: speech_scoring_model.pkl")
        print("✅ Scaler saved: feature_scaler.pkl")
        
        # 2. Save training predictions
        train_results_df = pd.DataFrame({
            'filename': self.train_filenames[:len(self.y_train)],
            'true_score': self.y_train,
            'predicted_score': self.train_predictions,
            'error': self.train_predictions - self.y_train,
            'abs_error': np.abs(self.train_predictions - self.y_train)
        })
        train_results_df.to_csv('train_results.csv', index=False)
        print("✅ Training results saved: train_results.csv")
        
        # 3. Save test predictions
        test_results_df = pd.DataFrame({
            'filename': self.test_filenames[:len(self.test_predictions)],
            'predicted_score': self.test_predictions
        })
        test_results_df.to_csv('test_predictions.csv', index=False)
        print("✅ Test predictions saved: test_predictions.csv")
        
        # 4. Save detailed report
        self.create_detailed_report()
        
        # 5. Save feature importance
        if self.model is not None:
            feature_names = [f'feature_{i}' for i in range(self.X_train.shape[1])]
            importances = self.model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            importance_df.to_csv('feature_importance.csv', index=False)
            print("✅ Feature importance saved: feature_importance.csv")
            
            # Plot feature importance
            plt.figure(figsize=(12, 6))
            top_20 = importance_df.head(20)
            plt.barh(range(len(top_20)), top_20['importance'].values)
            plt.yticks(range(len(top_20)), top_20['feature'].values)
            plt.xlabel('Importance')
            plt.title('Top 20 Most Important Features')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
            plt.show()
    
    def create_detailed_report(self):
        """Create a detailed text report of all results"""
        print("\n📄 Creating detailed report...")
        
        report = []
        report.append("=" * 70)
        report.append("SPEECH SCORING COMPETITION - COMPLETE RESULTS REPORT")
        report.append("=" * 70)
        
        # Dataset Information
        report.append("\n📁 DATASET INFORMATION")
        report.append("-" * 40)
        report.append(f"Training samples processed: {len(self.X_train)}")
        report.append(f"Test samples processed: {len(self.X_test)}")
        report.append(f"Number of features: {self.X_train.shape[1]}")
        
        # Model Performance
        report.append("\n🤖 MODEL PERFORMANCE")
        report.append("-" * 40)
        if 'cv_scores' in self.metrics:
            cv_scores = self.metrics['cv_scores']
            report.append(f"Cross-validation R² scores: {', '.join([f'{x:.4f}' for x in cv_scores])}")
            report.append(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        if 'train_r2' in self.metrics:
            report.append(f"\nTraining Set:")
            report.append(f"  R² Score: {self.metrics['train_r2']:.4f}")
            report.append(f"  RMSE: {self.metrics['train_rmse']:.4f}")
            report.append(f"  MAE: {self.metrics['train_mae']:.4f}")
        
        if 'val_r2' in self.metrics:
            report.append(f"\nValidation Set:")
            report.append(f"  R² Score: {self.metrics['val_r2']:.4f}")
            report.append(f"  RMSE: {self.metrics['val_rmse']:.4f}")
            report.append(f"  MAE: {self.metrics['val_mae']:.4f}")
        
        # Predictions Summary
        report.append("\n🔮 PREDICTIONS SUMMARY")
        report.append("-" * 40)
        
        if self.train_predictions is not None:
            train_errors = np.abs(self.train_predictions - self.y_train)
            report.append(f"\nTraining Predictions:")
            report.append(f"  Average Error: {train_errors.mean():.2f}")
            report.append(f"  Max Error: {train_errors.max():.2f}")
            report.append(f"  % within 5 points: {(train_errors <= 5).sum() / len(train_errors) * 100:.1f}%")
            report.append(f"  % within 10 points: {(train_errors <= 10).sum() / len(train_errors) * 100:.1f}%")
        
        if self.test_predictions is not None:
            report.append(f"\nTest Predictions:")
            report.append(f"  Count: {len(self.test_predictions)}")
            report.append(f"  Mean: {self.test_predictions.mean():.2f}")
            report.append(f"  Std: {self.test_predictions.std():.2f}")
            report.append(f"  Min: {self.test_predictions.min():.2f}")
            report.append(f"  Max: {self.test_predictions.max():.2f}")
            
            # Show first 10 predictions
            report.append(f"\n  First 10 predictions:")
            for i in range(min(10, len(self.test_predictions))):
                filename = self.test_filenames[i] if i < len(self.test_filenames) else f"test_{i}"
                report.append(f"    {filename}: {self.test_predictions[i]:.2f}")
        
        # Files Generated
        report.append("\n💾 OUTPUT FILES GENERATED")
        report.append("-" * 40)
        report.append("  1. submission.csv - Competition submission file")
        report.append("  2. train_results.csv - Training set predictions with errors")
        report.append("  3. test_predictions.csv - Test set predictions")
        report.append("  4. speech_scoring_model.pkl - Trained model")
        report.append("  5. feature_scaler.pkl - Feature scaler")
        report.append("  6. feature_importance.csv - Feature importance scores")
        report.append("  7. model_performance.png - Performance visualizations")
        report.append("  8. test_predictions_distribution.png - Test predictions distribution")
        report.append("  9. feature_importance.png - Feature importance plot")
        
        report.append("\n" + "=" * 70)
        report.append("END OF REPORT")
        report.append("=" * 70)
        
        # Save report
        with open('detailed_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        # Print report
        print('\n'.join(report[:50]))  # Print first 50 lines
        print("\n... (complete report saved to 'detailed_report.txt')")
        
        return report
    
    def run_full_pipeline(self):
        """Run the complete pipeline"""
        print("=" * 70)
        print("🚀 SPEECH SCORING COMPETITION - FULL PIPELINE")
        print("=" * 70)
        
        # Step 1: Validate structure
        if not self.validate_structure():
            return
        
        # Step 2: Load CSV data
        if not self.load_csv_data():
            return
        
        # Step 3: Process training data
        print("\n" + "=" * 50)
        print("PROCESSING TRAINING DATA")
        print("=" * 50)
        
        self.X_train, self.y_train, self.train_indices, self.train_filenames = self.process_dataset(
            self.train_df, self.train_audio_dir, is_training=True
        )
        
        if len(self.X_train) == 0:
            print("❌ No training data processed. Exiting.")
            return
        
        # Step 4: Process test data
        print("\n" + "=" * 50)
        print("PROCESSING TEST DATA")
        print("=" * 50)
        
        self.X_test, self.test_indices, self.test_filenames = self.process_dataset(
            self.test_df, self.test_audio_dir, is_training=False
        )
        
        if len(self.X_test) == 0:
            print("❌ No test data processed. Exiting.")
            return
        
        # Step 5: Train model
        print("\n" + "=" * 50)
        print("TRAINING MODEL")
        print("=" * 50)
        
        self.train_model(self.X_train, self.y_train)
        
        # Step 6: Make test predictions
        print("\n" + "=" * 50)
        print("MAKING PREDICTIONS")
        print("=" * 50)
        
        self.predict_test(self.X_test)
        
        # Step 7: Create visualizations
        print("\n" + "=" * 50)
        print("CREATING VISUALIZATIONS")
        print("=" * 50)
        
        self.create_visualizations(self.y_train, self.train_predictions)
        
        # Step 8: Create submission file
        print("\n" + "=" * 50)
        print("CREATING SUBMISSION")
        print("=" * 50)
        
        submission_file, submission_df = self.create_submission_file()
        
        # Step 9: Save all results
        print("\n" + "=" * 50)
        print("SAVING RESULTS")
        print("=" * 50)
        
        self.save_results()
        
        # Final summary
        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        print("\n🎯 NEXT STEPS:")
        print("   1. Upload 'submission.csv' to the competition platform")
        print("   2. Check 'detailed_report.txt' for complete analysis")
        print("   3. Review 'train_results.csv' for training performance")
        print("   4. View the visualization plots for insights")
        
        print("\n📁 All output files are ready in your project folder!")

# Main execution
if __name__ == "__main__":
    # Clear screen for better visibility
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("🎤 AUTOMATED SPEECH SCORING SYSTEM")
    print("=" * 60)
    print("Complete solution for speech scoring competition")
    print("Developed for: Voice Input → Score Calculation Pipeline")
    print("=" * 60)
    
    # Create solver instance
    solver = SpeechScoringCompetition("dataset")
    
    # Run the full pipeline
    try:
        solver.run_full_pipeline()
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Make sure your folder structure is correct")
        print("   2. Check that all .wav files are accessible")
        print("   3. Ensure CSV files have correct columns")
        print("   4. Try running with fewer features by modifying the code")
        
        # Try to save partial results
        try:
            if solver.model is not None:
                joblib.dump(solver.model, 'partial_model.pkl')
                print("   ✅ Partial model saved as 'partial_model.pkl'")
        except:
            pass