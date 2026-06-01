import os
import cv2
import numpy as np
import pandas as pd
import torch
import mediapipe as mp
from model import EmotionCNN

class EngagementPipeline:
    """
    Core pipeline to extract face presence, gaze proxy, and emotions from video frames,
    and compute a rule-based engagement score. Uses modern MediaPipe Tasks API.
    """
    def __init__(self, model_path=None, num_classes=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load custom Emotion CNN
        self.model = EmotionCNN(num_classes=num_classes)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded emotion model from {model_path}")
        else:
            print("Emotion model path not found. Running with uninitialized weights.")
        self.model.to(self.device)
        self.model.eval()

        # Emotion Labels (matching training)
        self.emotion_labels = ["Angry", "Bored", "Confused", "Happy", "Neutral"]
        
        # Initialize MediaPipe Tasks Face Landmarker
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Look for model in models/ folder
        task_path = os.path.join(os.path.dirname(__file__), "../models/face_landmarker.task")
        task_path = os.path.abspath(task_path)
        
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=task_path),
            running_mode=VisionRunningMode.IMAGE
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

        # Eye Landmark indices for gaze proxy (MediaPipe Face Mesh/Tasks indices)
        self.LEFT_EYE_OUTER = 33
        self.LEFT_EYE_INNER = 133
        self.RIGHT_EYE_INNER = 362
        self.RIGHT_EYE_OUTER = 263
        self.LEFT_PUPIL = 468
        self.RIGHT_PUPIL = 473

    def _estimate_gaze_proxy(self, face_lms, width, height):
        """
        Estimates gaze deviation based on pupil position relative to eye boundaries.
        Returns a normalized gaze deviation score (0 = looking center, 1 = looking far away).
        """
        try:
            # Helper to get coordinate tuple
            def get_pt(idx):
                lm = face_lms[idx]
                return np.array([lm.x * width, lm.y * height])

            # Left eye points
            l_outer = get_pt(self.LEFT_EYE_OUTER)
            l_inner = get_pt(self.LEFT_EYE_INNER)
            l_pupil = get_pt(self.LEFT_PUPIL)

            # Right eye points
            r_inner = get_pt(self.RIGHT_EYE_INNER)
            r_outer = get_pt(self.RIGHT_EYE_OUTER)
            r_pupil = get_pt(self.RIGHT_PUPIL)

            # Left eye midpoint & width
            l_mid = (l_outer + l_inner) / 2.0
            l_width = np.linalg.norm(l_outer - l_inner)
            l_dev = np.linalg.norm(l_pupil - l_mid) / (l_width + 1e-6)

            # Right eye midpoint & width
            r_mid = (r_inner + r_outer) / 2.0
            r_width = np.linalg.norm(r_outer - r_inner)
            r_dev = np.linalg.norm(r_pupil - r_mid) / (r_width + 1e-6)

            # Average deviation
            avg_dev = float((l_dev + r_dev) / 2.0)
            return min(avg_dev / 0.5, 1.0) # Normalise up to 0.5 deviation scale
        except Exception:
            return 0.5 # Default neutral fallback if landmark detection fails

    def _predict_emotion(self, frame, bbox):
        """
        Crops, processes, and classifies emotion of the detected face using custom CNN.
        """
        try:
            h, w, _ = frame.shape
            x, y, w_box, h_box = bbox
            # Clamp crop bounds
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + w_box), min(h, y + h_box)
            
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                return [0.0] * len(self.emotion_labels), "Neutral"

            # Preprocess crop: convert to gray, resize to 48x48, normalize
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48))
            img_tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(self.device)

            with torch.no_grad():
                logits = self.model(img_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            dominant_idx = np.argmax(probs)
            dominant_emotion = self.emotion_labels[dominant_idx]
            
            return probs.tolist(), dominant_emotion
        except Exception as e:
            # Fallback
            dummy_probs = [0.0] * len(self.emotion_labels)
            dummy_probs[-1] = 1.0 # 100% neutral
            return dummy_probs, "Neutral"

    def calculate_rule_score(self, face_present, gaze_deviation, dominant_emotion):
        """
        MSc Rule-based baseline: Computes an engagement score (0-100).
        """
        if not face_present:
            return 0.0

        # Emotion base score mapping
        emotion_scores = {
            "Happy": 95.0,
            "Neutral": 85.0,
            "Confused": 60.0,
            "Bored": 30.0,
            "Angry": 40.0
        }
        base_emotion_score = emotion_scores.get(dominant_emotion, 70.0)

        # Gaze penalty (looking away lowers score)
        # Deviation ranges from 0 (centered) to 1 (looking far away)
        gaze_penalty = gaze_deviation * 60.0 # Up to -60 penalty
        
        score = base_emotion_score - gaze_penalty
        return max(0.0, min(100.0, score))

    def process_video(self, video_path, max_frames=500):
        """
        Processes video frames and extracts timeline features for participants.
        Automatically tracks participants based on static grid locations.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 30.0 # Fallback
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        timeline_data = []
        frame_idx = 0

        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.landmarker.detect(mp_image)

            timestamp = frame_idx / fps
            detected_faces = []

            if results.face_landmarks:
                for idx, face_lms in enumerate(results.face_landmarks):
                    # Compute simple bounding box
                    xs = [lm.x for lm in face_lms]
                    ys = [lm.y for lm in face_lms]
                    x_min, x_max = min(xs) * width, max(xs) * width
                    y_min, y_max = min(ys) * height, max(ys) * height
                    
                    bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
                    center_x = x_min + (x_max - x_min) / 2
                    
                    gaze_dev = self._estimate_gaze_proxy(face_lms, width, height)
                    probs, dominant_emo = self._predict_emotion(frame, bbox)
                    
                    detected_faces.append({
                        "center_x": center_x,
                        "bbox": bbox,
                        "gaze_deviation": gaze_dev,
                        "emotion_probs": probs,
                        "dominant_emotion": dominant_emo
                    })

            # Sort detected faces from left to right (horizontal order) for static tracking
            detected_faces = sorted(detected_faces, key=lambda f: f["center_x"])

            # Save data for each participant
            # Assume a max of 4 participants. Map sorted detected faces to participant ID
            for p_idx in range(4):
                if p_idx < len(detected_faces):
                    face = detected_faces[p_idx]
                    rule_score = self.calculate_rule_score(True, face["gaze_deviation"], face["dominant_emotion"])
                    
                    record = {
                        "frame": frame_idx,
                        "timestamp": timestamp,
                        "participant_id": f"Participant_{p_idx + 1}",
                        "face_present": 1,
                        "gaze_deviation": face["gaze_deviation"],
                        "dominant_emotion": face["dominant_emotion"],
                        "engagement_score": rule_score
                    }
                    # Add individual emotion probabilities
                    for e_idx, label in enumerate(self.emotion_labels):
                        record[f"prob_{label}"] = face["emotion_probs"][e_idx]
                else:
                    # Participant missing in this frame
                    record = {
                        "frame": frame_idx,
                        "timestamp": timestamp,
                        "participant_id": f"Participant_{p_idx + 1}",
                        "face_present": 0,
                        "gaze_deviation": 1.0,
                        "dominant_emotion": "Absent",
                        "engagement_score": 0.0
                    }
                    for label in self.emotion_labels:
                        record[f"prob_{label}"] = 0.0

                timeline_data.append(record)

            frame_idx += 1

        cap.release()
        df = pd.DataFrame(timeline_data)
        
        # Temporal smoothing: Apply moving average to the engagement score
        if not df.empty:
            df["engagement_score_smoothed"] = df.groupby("participant_id")["engagement_score"].transform(
                lambda x: x.rolling(window=15, min_periods=1).mean()
            )

        return df

if __name__ == "__main__":
    # Test initialization
    pipeline = EngagementPipeline(num_classes=5)
    print("Pipeline successfully initialized with Tasks API.")
