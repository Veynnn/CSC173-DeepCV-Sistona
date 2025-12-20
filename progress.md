# CSC173 Deep Computer Vision Project Progress Report
**Student:** Lavigne Kaye S. Sistona, 2022-5619  
**Date:** December 15, 2023  
**Repository:** [https://github.com/LavigneSistona/CSC173-Project-Limpyo](https://github.com/LavigneSistona/CSC173-Project-Limpyo)  

## 📊 Current Status
| Milestone | Status | Notes |
|-----------|--------|-------|
| Dataset Preparation | ✅ Completed | 1000 images downloaded/preprocessed |
| Initial Training | ✅ In Progress | 15 epochs completed |
| Baseline Evaluation | ⏳ Pending | Training ongoing |
| Model Fine-tuning | ⏳ Not Started | Planned for tomorrow |

## 1. Dataset Progress
- **Total images:** 3000
- **Train/Test split:** 80%/20% (2400 train, 600 test)
- **Classes implemented:** 3 classes: Low Risk, Medium Risk, High Risk
- **Preprocessing applied:** Resize(128×128), normalization, augmentation (flip, rotate, brightness)

**Dataset preview:**
![Dataset Sample](data/Otoscopic_Data)

## 2. Training Progress
**Current Metrics:**
| Metric | Train | Val |
|--------|-------|-----|
| Loss | 0.312 | 0.399 |
| Accuracy | 89.6% | 85.3% |
| Precision | 0.892 | 0.847 |
| Recall | 0.901 | 0.856 |

## 3. Challenges Encountered & Solutions
| Issue | Status | Resolution |
|-------|--------|------------|
| Class imbalance in original data | ✅ Fixed | Mapped 5 medical classes → 3 balanced risk classes |
| File path errors on Windows | ✅ Fixed | Changed backslashes to forward slashes in paths |
| Training speed optimization | ✅ Fixed | Reduced image size to 128×128, simplified CNN architecture |
| Model overfitting | ⏳ Ongoing | Added dropout and batch normalization |

## 4. Next Steps (Before Final Submission)
- [ ] Complete training 
- [ ] Record 5-min demo video
- [ ] Write complete README.md with results