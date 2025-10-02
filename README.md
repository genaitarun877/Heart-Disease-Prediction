# ❤️ Heart Disease Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**An intelligent machine learning system to predict the risk of heart disease using clinical parameters**

[![Live Demo](https://img.shields.io/badge/Demo-Live%20on%20Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://heart-disease-prediction-tarun.streamlit.app/)

[Features](#-features) • [Live Demo](#-live-demo) • [Installation](#-installation) • [Usage](#-usage) • [Dataset](#-dataset) • [Model](#-model-performance)

</div>

---

## 📖 About The Project

Heart disease is one of the leading causes of death worldwide. Early detection and risk assessment can save lives. This project leverages **Machine Learning** to predict the likelihood of heart disease based on various clinical and demographic features.

The system includes:
- 📊 **Comprehensive EDA** - Detailed exploratory data analysis
- 🤖 **Multiple ML Models** - Comparison of 5 different algorithms
- 🎯 **Logistic Regression** - Selected as the final model
- 🌐 **Interactive Web App** - Built with Streamlit for easy predictions

---

## ✨ Features

- ✅ **Real-time Predictions** - Get instant heart disease risk assessment
- 📈 **Interactive UI** - User-friendly Streamlit interface
- 🔬 **Robust Preprocessing** - Handles missing values and outliers
- 📊 **Data Visualization** - Comprehensive EDA with beautiful plots
- 🎯 **High Accuracy** - Trained on 918 patient records (86.96% accuracy)
- 💾 **Model Persistence** - Saved models for quick deployment

---

## 🚀 Live Demo

### 🌐 Try it out now!

**[Click here to access the live application →](https://heart-disease-prediction-tarun.streamlit.app/)**

The application is deployed and running on Streamlit Cloud. You can test it immediately without any installation!

---

## 🎬 Demo

### Web Application Interface

The Streamlit app provides an intuitive interface where users can input their medical parameters and get instant predictions:

**Input Parameters:**
- Age (18-100 years)
- Sex (M/F)
- Chest Pain Type (ATA, NAP, TA, ASY)
- Resting Blood Pressure (mm Hg)
- Cholesterol (mg/dL)
- Fasting Blood Sugar > 120 mg/dL (Yes/No)
- Resting ECG (Normal, ST, LVH)
- Max Heart Rate
- Exercise-Induced Angina (Y/N)
- Oldpeak (ST Depression)
- ST Slope (Up, Flat, Down)

**Output:**
- ⚠️ High Risk of Heart Disease
- ✅ Low Risk of Heart Disease

---

## 🗂️ Dataset

The dataset contains **918 patient records** with **12 features**:

| Feature | Description | Type |
|---------|-------------|------|
| Age | Age of the patient | Numerical |
| Sex | Gender (M/F) | Categorical |
| ChestPainType | Type of chest pain (ATA, NAP, ASY, TA) | Categorical |
| RestingBP | Resting blood pressure (mm Hg) | Numerical |
| Cholesterol | Serum cholesterol (mg/dL) | Numerical |
| FastingBS | Fasting blood sugar > 120 mg/dL (1/0) | Categorical |
| RestingECG | Resting ECG results | Categorical |
| MaxHR | Maximum heart rate achieved | Numerical |
| ExerciseAngina | Exercise-induced angina (Y/N) | Categorical |
| Oldpeak | ST depression induced by exercise | Numerical |
| ST_Slope | Slope of the peak exercise ST segment | Categorical |
| HeartDisease | Target variable (1/0) | Categorical |

### Data Preprocessing Steps:
1. ✅ No null values found
2. 🔧 Replaced 0 values in Cholesterol with mean
3. 🔧 Replaced 0 values in RestingBP with mean
4. 🔄 One-hot encoding for categorical variables
5. 📏 StandardScaler for feature scaling

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step-by-step Installation

1. **Clone the repository**
```bash
git clone https://github.com/genaitarun877/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```

2. **Create a virtual environment** (Optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import streamlit; import pandas; import sklearn; print('All packages installed successfully!')"
```

---

## 💻 Usage

### Running the Streamlit App

1. Navigate to the project directory
```bash
cd "Sheeriyans ML/Heart IDisease"
```

2. Run the Streamlit app
```bash
streamlit run app.py
```

3. Open your browser and go to `http://localhost:8501`

4. Input the patient's medical parameters and click **Predict**

### Running the Jupyter Notebook

1. Launch Jupyter Notebook
```bash
jupyter notebook
```

2. Open `HeartDiesease.ipynb`

3. Run all cells to:
   - Perform exploratory data analysis
   - Train multiple models
   - Evaluate model performance
   - Save the best model

---

## 🎯 Model Performance

Multiple machine learning algorithms were evaluated on the test set:

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| **Logistic Regression** ⭐ | **86.96%** | **0.8857** |
| K-Nearest Neighbors (KNN) | 86.41% | 0.8815 |
| Naive Bayes | 85.33% | 0.8683 |
| Decision Tree | 78.26% | 0.8058 |
| Support Vector Machine (SVM) | 84.78% | 0.8679 |

**Final Model:** Logistic Regression was selected for deployment due to its:
- ✅ **Highest accuracy (86.96%)** and excellent F1 score (0.8857)
- ✅ Fast prediction time
- ✅ High interpretability for medical decisions
- ✅ Low computational requirements
- ✅ Balanced performance across precision and recall

---

## 📁 Project Structure

```
Heart IDisease/
│
├── app.py                    # Streamlit web application
├── HeartDiesease.ipynb       # Jupyter notebook with EDA and model training
├── heart.csv                 # Dataset
├── Logistic_heart.pkl        # Trained Logistic Regression model
├── scaler.pkl                # StandardScaler for feature scaling
├── columns.pkl               # Feature column names
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🛠️ Technologies Used

### Core Libraries
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Scikit-learn** - Machine learning algorithms
- **Joblib** - Model persistence

### Visualization
- **Matplotlib** - Static plotting
- **Seaborn** - Statistical visualizations

### Web Framework
- **Streamlit** - Interactive web application

### Analysis Tool
- **Sheryanalysis** - Custom analysis package

---

## 📊 Exploratory Data Analysis Highlights

The notebook includes comprehensive EDA with:

- 📈 **Distribution Analysis** - Histograms for Age, MaxHR, Cholesterol, RestingBP
- 📊 **Count Plots** - Sex, ChestPainType, FastingBS distributions
- 📦 **Box Plots** - Cholesterol vs HeartDisease
- 🎻 **Violin Plots** - Age vs HeartDisease
- 🔥 **Correlation Heatmap** - Feature relationships
- 🔍 **Outlier Detection** - Identifying and handling anomalies

---

## 🔮 Future Improvements

- [x] ~~Deploy on Streamlit Cloud~~ ✅ **COMPLETED!**
- [ ] Add more advanced models (XGBoost, Random Forest, Neural Networks)
- [ ] Implement SHAP values for model explainability
- [ ] Add prediction probability scores
- [ ] Create a REST API using Flask/FastAPI
- [ ] Deploy on additional cloud platforms (Heroku, AWS, Azure)
- [ ] Add user authentication and history tracking
- [ ] Implement cross-validation for more robust evaluation
- [ ] Create a mobile-friendly responsive design
- [ ] Add data visualization dashboard for patient history
- [ ] Implement batch prediction for multiple patients

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👨‍💻 Author

**Tarun Jaiswal**

- 🐙 GitHub: [@genaitarun877](https://github.com/genaitarun877)
- 💼 LinkedIn: [Tarun Jaiswal](https://www.linkedin.com/in/tarun-jaiswal911)
- 🌐 Live App: [Heart Disease Predictor](https://heart-disease-prediction-tarun.streamlit.app/)

---

## 📧 Contact

For any queries or suggestions, feel free to reach out:

- 📧 Email: [genai.tarun877@gmail.com](mailto:genai.tarun877@gmail.com)
- 💬 Issues: [GitHub Issues](https://github.com/genaitarun877/Heart-Disease-Prediction/issues)
- 💼 LinkedIn: [Connect with me](https://www.linkedin.com/in/tarun-jaiswal911)

---

## 🙏 Acknowledgments

- 📊 Dataset source: [Heart Failure Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- 🎓 [Sheeriyans Coding School](https://sheryians.com/) for guidance and support in Machine Learning
- 🌐 Streamlit Community for the amazing deployment platform
- 💻 The open-source community for incredible tools and libraries
- ❤️ All contributors and users who provide valuable feedback

---

<div align="center">

### ⭐ If you found this project helpful, please give it a star!

[![GitHub stars](https://img.shields.io/github/stars/genaitarun877/Heart-Disease-Prediction?style=social)](https://github.com/genaitarun877/Heart-Disease-Prediction/stargazers)

**Made with ❤️ by Tarun Jaiswal**

🔗 **Quick Links:** [Live App](https://heart-disease-prediction-tarun.streamlit.app/) | [GitHub](https://github.com/genaitarun877) | [LinkedIn](https://www.linkedin.com/in/tarun-jaiswal911)

</div>

