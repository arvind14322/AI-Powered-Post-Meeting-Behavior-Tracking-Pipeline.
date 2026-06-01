import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

class EngagementClassifier:
    """
    Trains and evaluates a machine learning model to classify engagement (Engaged vs. Disengaged)
    using engineered features from the video analysis pipeline.
    """
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_cols = [
            "face_present_mean",
            "gaze_dev_mean",
            "gaze_dev_var",
            "prob_Happy_mean",
            "prob_Neutral_mean",
            "prob_Confused_mean",
            "prob_Bored_mean",
            "prob_Angry_mean"
        ]

    def engineer_features(self, df, window_size_frames=30):
        """
        Groups frame-level data into rolling windows and extracts statistical features.
        """
        # Ensure sorting
        df = df.sort_values(by=["participant_id", "frame"])
        
        feature_list = []
        
        for pid, group in df.groupby("participant_id"):
            # Compute rolling features
            rolling = group.rolling(window=window_size_frames, min_periods=5)
            
            # Aggregate stats
            f_pres = rolling["face_present"].mean()
            g_mean = rolling["gaze_deviation"].mean()
            g_var = rolling["gaze_deviation"].var().fillna(0)
            
            p_happy = rolling["prob_Happy"].mean()
            p_neutral = rolling["prob_Neutral"].mean()
            p_conf = rolling["prob_Confused"].mean()
            p_bored = rolling["prob_Bored"].mean()
            p_angry = rolling["prob_Angry"].mean()
            
            # Target labels (simulated during training, predicted during inference)
            # In a real setup, these would be annotated. Here we engineer a proxy target for demonstration:
            # Let's say: High face presence + low gaze dev + high neutral/happy = Engaged (1), else Disengaged (0)
            is_engaged = (f_pres > 0.8) & (g_mean < 0.3) & ((p_happy + p_neutral) > 0.6)
            target = is_engaged.astype(int)
            
            temp_df = pd.DataFrame({
                "participant_id": group["participant_id"],
                "frame": group["frame"],
                "timestamp": group["timestamp"],
                "face_present_mean": f_pres,
                "gaze_dev_mean": g_mean,
                "gaze_dev_var": g_var,
                "prob_Happy_mean": p_happy,
                "prob_Neutral_mean": p_neutral,
                "prob_Confused_mean": p_conf,
                "prob_Bored_mean": p_bored,
                "prob_Angry_mean": p_angry,
                "label": target
            })
            
            feature_list.append(temp_df)
            
        result_df = pd.concat(feature_list).dropna()
        return result_df

    def train_model(self, X, y):
        """
        Trains a Random Forest classifier.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.model.fit(X_train, y_train)
        
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        report = classification_report(y_test, preds)
        
        print("Classifier Training Results:")
        print(f"Accuracy: {acc*100:.2f}%")
        print(f"ROC AUC: {auc:.4f}")
        print("\nClassification Report:\n", report)
        
        return acc, auc

    def save_model(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Classifier saved to {file_path}")

    def load_model(self, file_path):
        with open(file_path, "rb") as f:
            self.model = pickle.load(f)
        print(f"Classifier loaded from {file_path}")


def generate_synthetic_features(num_samples=1000):
    """
    Simulates pipeline output data to showcase training.
    """
    np.random.seed(42)
    data = []
    
    # 0 = Disengaged, 1 = Engaged
    labels = np.random.randint(0, 2, num_samples)
    
    for label in labels:
        if label == 1: # Engaged
            face_pres = np.random.uniform(0.9, 1.0)
            gaze_mean = np.random.uniform(0.05, 0.25)
            gaze_var = np.random.uniform(0.001, 0.02)
            p_happy = np.random.uniform(0.1, 0.4)
            p_neutral = np.random.uniform(0.5, 0.8)
            p_conf = np.random.uniform(0.0, 0.1)
            p_bored = np.random.uniform(0.0, 0.1)
            p_angry = np.random.uniform(0.0, 0.05)
        else: # Disengaged
            face_pres = np.random.choice([0.0, np.random.uniform(0.3, 0.8)], p=[0.3, 0.7])
            gaze_mean = np.random.uniform(0.35, 0.8) if face_pres > 0 else 1.0
            gaze_var = np.random.uniform(0.05, 0.2) if face_pres > 0 else 0.0
            p_happy = np.random.uniform(0.0, 0.1)
            p_neutral = np.random.uniform(0.1, 0.4)
            p_conf = np.random.uniform(0.2, 0.5)
            p_bored = np.random.uniform(0.3, 0.7)
            p_angry = np.random.uniform(0.1, 0.3)
            
        data.append([face_pres, gaze_mean, gaze_var, p_happy, p_neutral, p_conf, p_bored, p_angry, label])
        
    cols = [
        "face_present_mean", "gaze_dev_mean", "gaze_dev_var",
        "prob_Happy_mean", "prob_Neutral_mean", "prob_Confused_mean",
        "prob_Bored_mean", "prob_Angry_mean", "label"
    ]
    return pd.DataFrame(data, columns=cols)

if __name__ == "__main__":
    df_synthetic = generate_synthetic_features()
    X = df_synthetic.drop(columns=["label"])
    y = df_synthetic["label"]
    
    clf = EngagementClassifier()
    clf.train_model(X, y)
    clf.save_model("../models/engagement_classifier.pkl")
