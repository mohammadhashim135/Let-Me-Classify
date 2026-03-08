# **Let Me Classify** 🔍

A simple machine learning web application that classifies input text into categories such as **Technology, Sports, Politics, and Entertainment** using **Naive Bayes and TF-IDF vectorization**.

---

## **Overview**

**Let Me Classify** is a multi-category text classification project built using **Python and Streamlit**.
The application allows users to enter any piece of text and predicts the most relevant category.

The model processes the text using **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert words into numerical features.
It then uses a **Multinomial Naive Bayes classifier** to predict the category of the input text.

The app also displays the **probability distribution of all categories**, giving users insight into how confident the model is about its prediction.

---

## **Features** 🚀

* Multi-category text classification
* Interactive **Streamlit web interface**
* **TF-IDF vectorization** for text preprocessing
* **Naive Bayes classifier** for prediction
* Displays **prediction probabilities** with a visual bar chart
* Simple and lightweight ML project for beginners

---

## **Tech Stack** 🛠

* **Python**
* **Streamlit**
* **Scikit-learn**
* **Pandas**
* **TF-IDF Vectorizer**
* **Multinomial Naive Bayes**


---

## **Installation & Setup** 🏗

### **1. Clone the Repository**

```bash
git clone https://github.com/mohammadhashim135/Let-Me-Classify.git
cd Let-Me-Classify
```

### **2. Create a Virtual Environment**

```bash
python -m venv .venv
```

Activate the virtual environment:

**Windows**

```bash
.venv\Scripts\activate
```

**Mac / Linux**

```bash
source .venv/bin/activate
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Start the Application**

```bash
streamlit run app.py
```

---

## **Usage Guide** 📝

1. Open the Streamlit web interface in your browser.
2. Enter any text in the **input text box**.
3. Click the **Classify** button.
4. The application will:

   * Predict the **most relevant category** (Technology, Sports, Politics, Entertainment).
   * Display the **probability distribution** for each category using a bar chart.

---

## **Project Structure** 📂

```bash
let-me-classify/
│
├── app.py                # Streamlit web application
├── main.py               # Model training and evaluation
├── dataset.csv           # Dataset used for training
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
├── .gitignore
└── venv/                 # Virtual environment (not pushed to GitHub)
```

---


---
## **Contributing** 🤝
Contributions are welcome! If you’d like to improve feel free to fork the repo and submit a pull request.

### **Steps to Contribute:**

### **1. Fork the repository**

### **2. Create a new branch:**

```bash
git checkout -b feature-branch
```

### **3. Make your changes and commit:**

```bash
git commit -m "Added new feature"
```
### **4. Push to the branch:**

```bash
git push origin feature-branch
```
### **5. Open a Pull Request**
---
## **License** 📜
This project is licensed under the MIT License.

💡 Developed with ❤️ by [Mohammad Hashim](https://github.com/mohammadhashim135/Let-Me-Classify.git)

