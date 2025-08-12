# Laptop_Price_Prediction 

# 💻 Laptop Price Prediction  

This is a **Machine Learning project** that predicts the price of a laptop based on various hardware and software specifications like **Company, Type, RAM, Weight, Touchscreen, IPS Display, PPI, CPU brand, HDD, SSD, GPU brand, and Operating System**.  

🚀 **Live Demo:** [Laptop Price Prediction App](https://mridulmalvi-laptop-price-prediction-app-xvutei.streamlit.app/)  

---

## 📂 Project Structure
1. **.gitignore** – Specifies files/folders to be ignored by Git.  
2. **README.md** – Main documentation for the project.  
3. **requirements.txt** – List of Python dependencies needed for the project.  
4. **data/** – Folder containing datasets.  
    - `laptop_data.csv` – The original dataset used for training.  
5. **notebooks/** – Folder for Jupyter notebooks.  
    - `Laptop_Price_pred.ipynb` – Contains EDA and model training code.  
6. **models/** – Folder storing saved machine learning models.  
    - `df.pkl` – Preprocessed DataFrame object.  
    - `pipe.pkl` – Trained machine learning pipeline.  
7. **app/** – Folder containing deployment code.  
    - `app.py` – Streamlit app file for running the web application.  
8. **utils/** – Folder for helper functions (optional).  
    - `preprocess.py` – Functions for preprocessing and feature engineering.  

---

## 🛠 Features  
- **Data Preprocessing:** Cleaning & transforming raw data.  
- **Feature Engineering:** Extracting useful features such as PPI.  
- **Model Training:** Using machine learning algorithms to predict prices.  
- **Streamlit Deployment:** Interactive UI for real-time predictions.  
- **Pickle Files:** Storing trained model and processed data for fast loading.  

---

## 📊 Dataset Features  
| Feature       | Description |
|--------------|-------------|
| Company      | Laptop manufacturer brand |
| TypeName     | Type of laptop (Ultrabook, Gaming, etc.) |
| Ram          | RAM size in GB |
| Weight       | Weight in kg |
| Touchscreen  | 1 if touchscreen, else 0 |
| IPS          | 1 if IPS display, else 0 |
| PPI          | Pixels Per Inch of display |
| Cpubrand     | CPU brand (Intel, AMD, etc.) |
| HDD          | Hard Disk capacity in GB |
| SSD          | Solid State Drive capacity in GB |
| Gpu brand    | GPU brand |
| os           | Operating System |

---

## 📜 License
This project is open-source and available under the MIT License.

---


## ⚙️ Installation & Usage  

1. **Clone the repository**  
```bash
git clone https://github.com/<your-username>/laptop-price-prediction.git
cd laptop-price-prediction

#Install dependencies
pip install -r requirements.txt

#Run the Streamlit app
streamlit run app.py

#Or use the hosted version
[Laptop Price Prediction App](https://mridulmalvi-laptop-price-prediction-app-xvutei.streamlit.app/)


#Technologies Used
Python 🐍
Pandas, NumPy – Data manipulation
Scikit-learn – Machine learning model
Streamlit – Web app deployment
Pickle – Model serialization

---

