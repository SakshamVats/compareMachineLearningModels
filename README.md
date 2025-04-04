# 🔍 Compare Scikit-Learn Models: Performance Benchmarking  

## 🚀 Overview  

This project compares multiple **machine learning models** on a classification dataset using **Scikit-Learn**.  
It evaluates **logistic regression, decision trees, random forests, SVM, KNN, Naïve Bayes, and neural networks** based on:  
✅ **Cross-validation accuracy**  
✅ **Test set accuracy**  
✅ **Performance visualization**  

---

## 📌 Features  

- **Preprocessing:** Automatic dataset handling and feature engineering  
- **Modular Design:** Each model runs independently with structured training & evaluation  
- **Hyperparameter Support:** Tune key parameters for sensitive models  
- **Visualization:** Compare model accuracies with a stylish bar chart  

---

## ⚡ Models Implemented  

| Model                | Cross-Val Accuracy | Test Accuracy |
|----------------------|-------------------|--------------|
| Logistic Regression | ✅ Included       | ✅ Included |
| Decision Tree       | ✅ Included       | ✅ Included |
| Random Forest      | ✅ Included       | ✅ Included |
| SVM                | ✅ Included       | ✅ Included |
| KNN                | ✅ Included       | ✅ Included |
| Naïve Bayes        | ✅ Included       | ✅ Included |
| Neural Network     | ✅ Included       | ✅ Included |

---

## 🔧 Installation  

### **1️⃣ Clone the repository**  
```bash
git clone https://github.com/yourusername/compare_sklearn_models.git
cd compare_sklearn_models
```

### **2️⃣ Install dependencies**  
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the project**  
```bash
python main.py
```

---

## 📊 Results  

The project outputs:  
- **Model accuracies on test & validation sets**  
- **Performance bar chart comparing models**  

## 📂 Project Structure  

```
compare_sklearn_models/
│── data/
│   ├── get_dataset.py      # Downloads & loads the dataset
│   ├── __init__.py
│── models/
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── svm.py
│   ├── knn.py
│   ├── naive_bayes.py
│   ├── neural_net.py
│   ├── __init__.py
│── utils/
│   ├── train_model.py       # Handles model training
│   ├── evaluate_model.py    # Evaluates models & prints results
│   ├── preprocess.py        # Data preprocessing pipeline
│   ├── __init__.py
│── main.py                 # Runs all models & visualizes results
│── README.md
```

---

## 🤝 Contributing  

Contributions are welcome! If you want to:  
- Improve model training strategies  
- Add more ML models  
- Enhance visualization  

Feel free to open an **issue** or submit a **pull request**! 🎉  
##

🌟 **If you like this project, don't forget to star ⭐ the repository!**  
