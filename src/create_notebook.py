import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Video Meeting Engagement Analyzer\n",
    "\n",
    "This notebook demonstrates the pipeline for estimating engagement in recorded video meetings. It is designed as an MSc Data Science student portfolio project.\n",
    "\n",
    "## Pipeline Architecture\n",
    "1. **Facial Emotion Recognition (FER)**: A custom VGG-like CNN trained on emotion expressions.\n",
    "2. **Feature Extraction**: MediaPipe Face Mesh detects faces, calculates eye landmarks, and computes a gaze proxy.\n",
    "3. **Engagement Classifier**: A Random Forest classifier trained on frame-level features (presence, gaze deviation, emotions).\n",
    "4. **Timeline Analysis**: Visualization of smoothed engagement trends and flagged periods of disengagement."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. Imports and Setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import os\n",
    "sys.path.append('../src')\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import torch\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from model import EmotionCNN\n",
    "from train import SyntheticEmotionDataset\n",
    "from classifier import EngagementClassifier, generate_synthetic_features\n",
    "\n",
    "print(\"Setup complete.\")\n",
    "print(f\"CUDA Available: {torch.cuda.is_available()}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Custom Emotion CNN Training\n",
    "We train the custom PyTorch VGG-like CNN on emotion data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from torch.utils.data import DataLoader\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "\n",
    "# Hyperparameters\n",
    "epochs = 3\n",
    "batch_size = 32\n",
    "num_classes = 5\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "\n",
    "# Datasets\n",
    "train_ds = SyntheticEmotionDataset(num_samples=500, num_classes=num_classes)\n",
    "val_ds = SyntheticEmotionDataset(num_samples=100, num_classes=num_classes)\n",
    "\n",
    "train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)\n",
    "val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)\n",
    "\n",
    "model = EmotionCNN(num_classes=num_classes).to(device)\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "optimizer = optim.Adam(model.parameters(), lr=0.001)\n",
    "\n",
    "# Simple validation training run\n",
    "for epoch in range(1, epochs + 1):\n",
    "    model.train()\n",
    "    for imgs, lbls in train_loader:\n",
    "        imgs, lbls = imgs.to(device), lbls.to(device)\n",
    "        optimizer.zero_grad()\n",
    "        loss = criterion(model(imgs), lbls)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "    print(f\"Epoch {epoch} finished.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Machine Learning Engagement Classifier\n",
    "We engineer rolling window statistical features and train an engagement classifier (Random Forest)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Simulate pipeline output features\n",
    "df_features = generate_synthetic_features(n_samples := 800)\n",
    "\n",
    "X = df_features.drop(columns=[\"label\"])\n",
    "y = df_features[\"label\"]\n",
    "\n",
    "clf = EngagementClassifier()\n",
    "acc, auc = clf.train_model(X, y)\n",
    "print(f\"Model performance: Accuracy = {acc*100:.2f}%, AUC = {auc:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. Visualizing Participant Engagement Over Time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create synthetic meeting timeline data for 2 participants over 200 frames\n",
    "np.random.seed(10)\n",
    "frames = np.arange(200)\n",
    "timestamps = frames / 10.0\n",
    "\n",
    "# Participant 1 (Attentive)\n",
    "score1 = 80 + np.random.normal(0, 5, 200)\n",
    "score1 = np.clip(score1, 0, 100)\n",
    "\n",
    "# Participant 2 (Disengaged halfway)\n",
    "score2 = 85 + np.random.normal(0, 5, 200)\n",
    "score2[90:150] -= 50 # Disengagement drop\n",
    "score2 = np.clip(score2, 0, 100)\n",
    "\n",
    "# Smooth using rolling window\n",
    "smoothed1 = pd.Series(score1).rolling(window=10, min_periods=1).mean()\n",
    "smoothed2 = pd.Series(score2).rolling(window=10, min_periods=1).mean()\n",
    "\n",
    "plt.figure(figsize=(12, 5))\n",
    "plt.plot(timestamps, smoothed1, label=\"Participant 1 (Attentive)\", color=\"#4CAF50\", linewidth=2)\n",
    "plt.plot(timestamps, smoothed2, label=\"Participant 2 (Losing Focus)\", color=\"#F44336\", linewidth=2)\n",
    "plt.axhspan(0, 45, color='#FFCDD2', alpha=0.3, label='Disengagement Zone')\n",
    "plt.title(\"Engagement Score Timeline Over Meeting Session\", fontsize=14)\n",
    "plt.xlabel(\"Time (Seconds)\", fontsize=12)\n",
    "plt.ylabel(\"Engagement Score (0 - 100)\", fontsize=12)\n",
    "plt.legend(loc=\"lower left\")\n",
    "plt.grid(True, linestyle=\"--\", alpha=0.5)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Ethical Grounding & Limitations\n",
    "\n",
    "For academic rigor, it's critical to state that:\n",
    "- **Expression ≠ Intent**: Face expression does not directly measure cognitive focus or comprehension.\n",
    "- **Bias & Fairness**: Facial landmark estimators can exhibit demographic performance bias (skin tone, lighting, head angle).\n",
    "- **Consent & Transparency**: Real-world application of such models requires explicit consent, transparency, and data privacy protocols."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

os.makedirs("../notebooks", exist_ok=True)
with open("../notebooks/engagement_analysis.ipynb", "w") as f:
    json.dump(notebook_content, f, indent=1)

print("Notebook generated successfully.")
