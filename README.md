# 📊 Coupon Purchase Prediction  

Forecast customer coupon redemption with AI insights.  
Optimize your marketing campaigns with **data-driven predictions powered by LightGBM, FastAPI & a modern frontend**.  

---

## 🚀 Project Overview
This project predicts whether a **customer will redeem a coupon** in a campaign using ML techniques.  
It integrates:
- **LightGBM** for high-performance modeling  
- **SMOTE & balancing** for handling class imbalance  
- **FastAPI backend** exposing prediction API  
- **Modern frontend (TailwindCSS + React)** for interactive visualization  

---

## 📂 Dataset  
We used [Predicting Coupon Redemption Dataset](https://www.kaggle.com/datasets/vasudeva009/predicting-coupon-redemption) which includes:
- `train.csv`: Historical coupon usage with redemption labels  
- `test.csv`: Evaluation dataset  
- `customer_demographics.csv`: Age, marital status, family size, income bracket  
- `customer_transactions.csv`: Past purchases, quantity, discounts, recency  
- `campaign_data.csv`: Campaign duration and type  

---

## 🧠 Models Used  
We tested multiple models:  
- **Logistic Regression** (baseline)  
- **Random Forest Classifier**  
- **LightGBM (Final Model)** → Achieved **Val AUC: 0.925**  

---

## ⚙️ Tech Stack  
- **Python**: Pandas, NumPy, Scikit-learn, LightGBM  
- **Backend**: FastAPI + Uvicorn  
- **Frontend**: React + TailwindCSS (via [Lovable](https://lovable.dev))  
- **Deployment**: Ngrok (demo), Netlify/Vercel (frontend)  

---

## 📊 Results
- **Final Validation AUC**: `0.925`  
- **Class 1 Recall improved** with customer demographics + transactions features  
- Interactive website allows live predictions  

---

Ahh I see what happened 🙌
The numbers in your **headings (`###2️⃣`, `###3️⃣`)** don’t have spaces, so GitHub doesn’t render them correctly.

Here’s the **fixed version** of your `README.md` section with proper spacing so it shows up neat:


## 🔧 Setup & Installation

### 1️⃣ Clone the repo
```bash
git clone https://github.com/Yashwant1105/coupon-price-prediction.git
cd coupon-price-prediction
````

### 2️⃣ Setup environment

```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 3️⃣ Run FastAPI backend

```bash
uvicorn src.app:app --reload
```

API runs at:
👉 [http://127.0.0.1:8000](http://127.0.0.1:8000)
👉 Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 4️⃣ Frontend

Frontend (React + Tailwind) deployed separately via **Netlify/Vercel**.
Update `API_URL` inside frontend code to your **ngrok/production URL**.



📌 API Usage

POST /predict
```
{
  "customer_id": 1059,
  "campaign_id": 18,
  "coupon_id": 1060
}
```

Response
```
{
  "purchase_probability": 0.78,
  "recommendation": "Likely to purchase ✅"
}
```

🌟 Demo

🔗 Live Frontend: https://couponai.lovable.app/

🔗 Backend (ngrok): Run `ngrok http 8000` → use the generated URL (e.g., `https://xxxx-xx-xx-xx.ngrok-free.app`)  

👥 Team

Built by Team X using LightGBM + FastAPI + Tailwind.

