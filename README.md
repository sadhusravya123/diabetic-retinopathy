# Explainable AI for Diabetic Retinopathy Screening

##  Project Overview
This project uses Deep Learning to classify Diabetic Retinopathy (DR) severity from retinal fundus images. It compares three models and uses Grad-CAM to visualize the regions influencing model predictions.

##  Objectives
- Classify retinal images into five DR severity levels.
- Compare a custom CNN, EfficientNetB0, and ResNet50.
- Evaluate models using Accuracy, Precision, Recall, Macro-F1, and Quadratic Weighted Kappa (QWK).
- Visualize model predictions using Grad-CAM.
- Flag low-confidence predictions for additional review.

##  Dataset
**APTOS 2019 Blindness Detection Dataset**

The dataset contains 3,662 labelled retinal images divided into five classes:

| Label | DR Severity |
|---|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

Dataset split:
- Training: 70%
- Validation: 15%
- Testing: 15%

## 🛠️ Technologies Used
- Python
- PyTorch
- Torchvision
- NumPy and Pandas
- Pillow (PIL)
- Scikit-learn
- Matplotlib and Seaborn
- Google Colab with GPU acceleration

##  Models Implemented
1. **Custom CNN:** Baseline model trained from scratch.
2. **EfficientNetB0:** Transfer learning followed by fine-tuning.
3. **ResNet50:** Transfer learning with a frozen feature extractor.

## Methodology
1. Load and validate the dataset.
2. Resize images to 224 × 224 pixels.
3. Split the data using stratified sampling.
4. Apply image transformations and normalization.
5. Train the models using class-weighted cross-entropy loss.
6. Evaluate model performance on the test set.
7. Generate confusion matrices and performance comparisons.
8. Apply Grad-CAM for visual explanations.
9. Flag predictions with confidence below 0.60 for review.

## Results

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Custom CNN | 0.67 | 0.45 |
| EfficientNetB0 (Fine-tuned) | 0.73 | 0.58 |
| ResNet50 | 0.68 | 0.49 |

**Best-performing model:** Fine-tuned EfficientNetB0, achieving 73% test accuracy and a 0.58 Macro-F1 score.

##  How to Run
1. Open the project notebook in Google Colab.
2. Download the APTOS 2019 Blindness Detection dataset from Kaggle.
3. Upload or mount the dataset in the Colab environment.
4. Update the dataset paths if required.
5. Run the notebook cells sequentially.
6. Review the model metrics, confusion matrices, training curves, and Grad-CAM visualizations.

##Future Enhancements
- Explore Vision Transformers and hybrid architectures.
- Improve confidence calibration and uncertainty estimation.
- Incorporate lesion-level segmentation.
- Validate models on independent retinal image datasets.
- Evaluate generalization across different cameras and patient populations.

