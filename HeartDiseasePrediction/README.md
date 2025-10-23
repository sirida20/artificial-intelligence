# Heart Disease Risk Prediction (Flask + HTML)

## Overview

This project is an end-to-end demonstration of deploying a **Machine Learning model** as a web service.

* **Goal:** Predict the risk of heart disease (0: Low Risk / 1: High Risk) based on 13 clinical features.

* **Backend:** Uses **Python (Flask)** to train a Random Forest Classifier on `heart_disease_cleaned.csv`, preprocess input data, and serve predictions via a REST API.

* **Frontend:** A simple HTML/JavaScript interface that collects user input and communicates with the Flask server.

* **Dataset Source:** The data used for training is based on the Heart Disease Dataset available on [Kaggle](https://www.kaggle.com/datasets/abdmental01/heart-disease-dataset).

---

## How to Run Locally

### Prerequisites

You must have **Python 3** installed. Install the necessary libraries using `pip`:

```bash
pip install pandas numpy scikit-learn flask flask-cors
```

**1. Setup**

Ensure the following three files are in the same folder:

i) `app.py`

ii) `prediction_client.html`

iii) `heart_disease_cleaned.csv`

**2. Start the Server**
   
Open your terminal, navigate to the project directory, and run the server:

Navigate to the directory

```bash
cd /path/to/your/project/folder
```

Start the Flask server

```bash
py -3 app.py
```
Note: The command might be `python3 app.py` or `python app.py` depending on your setup. Keep this terminal open!

**3. Open the Client**
   
i) Locate `prediction_client.html` in your file explorer.

ii) Double-click the file to open it in your browser.

iii) Enter the data and test the prediction!

---

## License
This project is released under the MIT License.
