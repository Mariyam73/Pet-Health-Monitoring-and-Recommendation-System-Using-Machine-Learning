# Pet Health Monitoring and Recommendation System Using Machine Learning

This project is a machine learning-based web application that predicts a pet's health status (Healthy/Unhealthy) and provides personalised recommendations based on behavioural clustering.

🔗 **Project Repository:**  
[GitHub Repo](https://github.com/Mariyam73/Pet-Health-Monitoring-and-Recommendation-System-Using-Machine-Learning/tree/main)

---

## Repository Structure

```plaintext
Pet-Health-Monitoring-and-Recommendation-System-Using-Machine-Learning/
│
├── Codes/                                   # Contains all ML code
│   ├── Building and Testing Codes/          # Experimental/test files
│   └── Pet Health Model Codes          # Main build-and-run script
│
├── Data/                      # Dataset and preprocessed files
│   ├── synthetic_dog_breed_health_data.csv
│   └── Dog_Health_Preprocessed.csv
│
├── Models/                    # Pre-trained ML models
│   ├── random_forest.pkl
│   ├── kmeans.pkl
│   └── scaler.pkl
│
├── pet-health-app/           # Web application files
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── requirements.txt
│
└── README.md                  # You're here!

```

---

## ⚙️ Setup Instructions

Follow these steps to get the project running locally on your system.

---

### ✅ 1. Clone the Repository

```bash
git clone https://github.com/Mariyam73/Pet-Health-Monitoring-and-Recommendation-System-Using-Machine-Learning.git
cd Pet-Health-Monitoring-and-Recommendation-System-Using-Machine-Learning
```

---

### ✅ 2. Create and Activate a Virtual Environment

Creating a virtual environment ensures package dependencies are isolated from your global Python setup.

#### Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### ✅ 3. Install Required Libraries

Navigate to the app directory:

```bash
cd "Health app"
```

Install dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

If the file is missing, you can install manually:

```bash
pip install flask pandas scikit-learn numpy
```

---

### ✅ 4. Run the Web Application

Make sure you're in the `Health app` directory, then run:

```bash
python app.py
```

After starting, open your browser and go to:

```
http://127.0.0.1:5000/
```

You’ll see the web interface rendered using `templates/index.html`.

---

## 🧠 Model Training and Testing

If you want to build or retrain models:

- Use `Codes/` to train or update models.
- The trained model will be stored in the `Models/` directory.
- You can test or experiment using `Codes/testing.ipynb`.

---

## 💡 Tech Stack

- **Python 3.7+**
- **Flask** – For building the web interface
- **Scikit-learn** – For ML algorithms
- **Pandas, NumPy** – For data manipulation
- **HTML + Jinja2 Templates** – For frontend rendering

---

## 📝 Notes

- Make sure port **5000** is not blocked before running Flask.
- Always activate the virtual environment before running the app.
- Do not rename or move files/folders unless you update all references.

---

## 📬 Contact

For issues or feedback, raise an issue on the GitHub repo:

👉 [GitHub Issues](https://github.com/Mariyam73/Pet-Health-Monitoring-and-Recommendation-System-Using-Machine-Learning/issues)

---


