# ML Model Implementations & Assignments 🎓

A curated collection of machine-learning implementations — in Python and R — covering core algorithms, ensemble methods, and a text-classification pipeline.  
Ideal for learning, teaching, benchmarking, or as a solid base for further experimentation.

---

## 🚀 Why This Project

- **Educational clarity:** Clean, self-contained implementations of common ML algorithms — great for learning or teaching ML fundamentals.  
- **Diverse methods:** Includes neural networks, classical classifiers, ensemble learning, and text-based Naive Bayes.  
- **Hands-on workflows:** Real data pipelines (preprocessing → training → evaluation) for both tabular and text data.  
- **Flexible for experimentation:** Easily modifiable to test new models, data-sets, or preprocessing schemes.

---


> ✏️  If file or folder names differ locally, adjust accordingly.

---

## 🧠 Included Models & Scripts

### **Neural Network / CNN Classifier (Python)**  
**File:** `CNN_NNimplementation.py`  
A ready-to-run neural network (or convolutional network) classifier, covering data loading, model definition, training loops, and evaluation.

### **Decision Tree Classifier (Python)**  
**File:** `DTClassifier_implementation`  
Classic decision-tree classifier implementation: training, prediction, and evaluation on test data.  

### **Ensemble Classifier with Stacking (R)**  
**File:** `ensemble_classifier.R`  
Trains multiple models (Random Forest, AdaBoost, Logistic Regression, Decision Tree, k-NN, Naive Bayes, Neural Net) using cross-validation, then combines them using stacking via `caret` + `caretEnsemble`. Final evaluation and performance output are saved to `Output_final_April18_2019.txt`.  

### **Naive Bayes Text Classifier — 20 Newsgroups (R)**  
**File:** `nb_20newsgroups.R`  
Implements a **multinomial Naive Bayes** classifier (both MLE & Bayesian with Laplace smoothing) on document-word data from 20 Newsgroups.  
Performs likelihood estimation, class prior computation, log-likelihood scoring, predictions (MLE & Bayesian), and outputs confusion matrices and accuracy metrics on training and test sets.

---

## 💻 Getting Started

### Clone the repository

```bash
git clone https://github.com/sayaneshome/ML_model_implementations.git
cd ML_model_implementations
