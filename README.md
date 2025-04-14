# 🧠 Decoding Motor Intention from Neural Signals Using a Simple Neural Network

This project decodes motor intention—specifically, trial success—using extracellular spike recordings from the primary motor cortex (M1) of a macaque performing a sequential reaching task. The decoding model is built using a simple neural network implemented in PyTorch, all within a single Jupyter notebook.

---

## 📁 Dataset

**Source**: [CRCNS PMd-1 dataset](https://crcns.org/data-sets/movements/pmd-1/about-pmd-1)  
**File used**: `MT_S1_raw.mat`

### Dataset Contents
- `M1`: Spike-sorted unit activity (spike times, waveform, unit IDs)
- `trial_table`: Metadata for each trial, including target timings and trial outcomes
- **Trial label**: Column 23 in `trial_table` marks trial result  
  - `82` = success (rewarded)  
  - `70` = failure  
  - Other values are excluded

---

## 📒 Notebook Overview: `MT_S1_train.ipynb`

### Pipeline Summary
1. **Spike Count Extraction**
   - Spike times for each unit are binned in a 500ms window before the go cue.
2. **Label Cleaning**
   - Trials with successful (`82`) and failed (`70`) outcomes are used for binary classification.
3. **Preprocessing**
   - Trials with invalid or missing labels are dropped.
   - Spike counts are standardized with `StandardScaler`.
4. **Neural Network**
   - 2-layer MLP using `torch.nn.Sequential`
   - Binary cross-entropy loss with `Adam` optimizer
5. **Evaluation**
   - Accuracy and ROC-AUC score
   - Confusion matrix and ROC curve visualizations

---

## 📈 Results

- **Accuracy**: ~90%
- **AUC**: >0.90
- The model demonstrates effective decoding of trial outcomes based on pre-movement spiking activity.

---

## 🧰 Dependencies

Run in a Python 3.8+ environment with:

```bash
pip install numpy scipy matplotlib scikit-learn torch
