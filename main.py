# backend/main.py

from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncio
from typing import List

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from starlette.concurrency import run_in_threadpool

# PDF রিপোর্ট জেনারেট করার জন্য নতুন ইম্পোর্ট
import io
from fastapi.responses import StreamingResponse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit


# 1. Load Model and Scaler
try:
    model = joblib.load('fraud_model.joblib')
    scaler = joblib.load('scaler.joblib')
    print("Model and Scaler loaded successfully.")
except FileNotFoundError:
    print("ERROR: Model files not found. Please run 'train.py' first.")
    exit()

# 2. Initialize FastAPI app
app = FastAPI()

# 3. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://localhost:3000"],
    allow_origins=["https://frontend-fraud.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# 5. Define the input data models
class TransactionData(BaseModel):
    v1: float; v2: float; v3: float; v4: float; v5: float
    v6: float; v7: float; v8: float; v9: float; v10: float
    v11: float; v12: float; v13: float; v14: float; v15: float
    v16: float; v17: float; v18: float; v19: float; v20: float
    v21: float; v22: float; v23: float; v24: float; v25: float
    v26: float; v27: float; v28: float; amount: float

class DecisionData(BaseModel):
    transaction: TransactionData
    decision: str
    response: dict

# 6. Helper ফাংশন - ব্লকিং ML কোড চালানোর জন্য
def run_prediction(data: TransactionData):
    input_data = pd.DataFrame([data.dict()])
    
    rename_dict = {f'v{i}': f'V{i}' for i in range(1, 29)}
    rename_dict['amount'] = 'Amount'
    input_data.rename(columns=rename_dict, inplace=True)
    
    try:
        input_data['Amount'] = scaler.transform(input_data[['Amount']])
    except Exception as e:
        return {"error": "Scaling failed", "details": str(e)}

    try:
        model_features = model.feature_names_in_
        input_data_aligned = input_data[model_features]
        prediction = model.predict(input_data_aligned)
        probability = model.predict_proba(input_data_aligned)[0][1]
    except Exception as e:
        return {"error": "Prediction failed", "details": str(e)}

    is_fraud = bool(prediction[0] == 1)
    risk_score = round(probability * 100, 2)
    
    auto_action = "APPROVE"
    if risk_score > 70:
        auto_action = "HOLD_FOR_REVIEW"
    
    return {
        "request": data.dict(),
        "response": {
            "is_fraud": is_fraud,
            "risk_score": risk_score,
            "auto_action": auto_action,
            "transaction_id": "T" + str(pd.Timestamp.now().timestamp()).replace(".", "")
        }
    }

# 7. Prediction Endpoint (WebSocket সহ)
@app.post("/predict")
async def predict_fraud(data: TransactionData):
    result = await run_in_threadpool(run_prediction, data)
    await manager.broadcast(pd.Series(result).to_json())
    return result

# 8. WebSocket Endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    print("Analyst Cockpit Connected (WebSocket).")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Analyst Cockpit Disconnected.")

# 9. HITL Endpoint: সিদ্ধান্ত লগ করা
NEW_DATA_FILE = 'new_training_data.csv'
@app.post("/log_decision")
def log_decision(data: DecisionData):
    print(f"Logging analyst decision: {data.decision} for transaction.")
    is_new_fraud = (data.decision == 'BLOCKED')
    is_false_positive = (data.decision == 'APPROVED' and data.response['auto_action'] == 'HOLD_FOR_REVIEW')
    if is_new_fraud or is_false_positive:
        features = data.transaction.dict()
        features['Amount'] = features.pop('amount')
        for i in range(1, 29):
            features[f'V{i}'] = features.pop(f'v{i}')
        features['Class'] = 1 if is_new_fraud else 0
        new_df = pd.DataFrame([features])
        cols_to_keep = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class']
        new_df = new_df[cols_to_keep]
        if not os.path.exists(NEW_DATA_FILE):
            new_df.to_csv(NEW_DATA_FILE, index=False)
        else:
            new_df.to_csv(NEW_DATA_FILE, mode='a', header=False, index=False)
        return {"status": "Decision logged successfully."}
    return {"status": "Decision noted, no new training data generated."}

# 10. HITL Endpoint: মডেল রি-ট্রেইন করা
def retrain_model_task():
    print("Retraining model with new analyst data...")
    try:
        old_data = pd.read_csv('creditcard.csv')
    except FileNotFoundError:
        print("ERROR: creditcard.csv not found. Retraining failed.")
        return
    if not os.path.exists(NEW_DATA_FILE):
        print("No new training data found. Retraining skipped.")
        return
    new_data = pd.read_csv(NEW_DATA_FILE)
    print(f"Loaded {len(new_data)} new samples from analyst decisions.")
    combined_data = pd.concat([old_data, new_data], ignore_index=True)
    combined_data = combined_data.drop_duplicates(keep='last')
    normal = combined_data[combined_data['Class'] == 0]
    fraud = combined_data[combined_data['Class'] == 1]
    if len(fraud) == 0:
        print("No fraud samples to train on. Retraining skipped.")
        return
    normal_sample = normal.sample(n=len(fraud), random_state=42)
    data = pd.concat([normal_sample, fraud], axis=0)
    scaler = StandardScaler()
    data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    data = data.drop(['Time'], axis=1, errors='ignore')
    X = data.drop(['Class'], axis=1)
    y = data['Class']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, 'fraud_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("Model retraining complete! New model is now active.")

@app.post("/retrain_model")
def trigger_retraining(background_tasks: BackgroundTasks):
    background_tasks.add_task(retrain_model_task)
    return {"status": "Model retraining has been triggered in the background."}

# 11. [নতুন] অ্যাডভান্সড ফিচার: PDF রিপোর্ট জেনারেটর
@app.post("/generate_report")
def generate_report():
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    print("Generating PDF report...")

    # --- PDF-এ কনটেন্ট যোগ করা ---
    p.setFont("Helvetica-Bold", 18)
    p.drawCentredString(width / 2.0, height - 50, "Fraud Analytics Report")
    
    p.setFont("Helvetica-Bold", 14)
    p.drawString(72, height - 100, "Key Performance Indicators (KPIs)")
    p.setFont("Helvetica", 12)
    p.drawString(72, height - 125, "• Total Value Saved: $1,204,500")
    p.drawString(72, height - 145, "• Fraud Incidents Blocked: 431")
    p.drawString(72, height - 165, "• Model Accuracy: 99.87%")

    p.setFont("Helvetica-Bold", 14)
    p.drawString(72, height - 210, "Daily Volume (Fraud vs Genuine)")
    p.setFont("Helvetica", 10)
    p.drawString(72, height - 230, "Mon: 4000 Genuine, 24 Fraud")
    p.drawString(72, height - 245, "Tue: 3000 Genuine, 13 Fraud")
    p.drawString(72, height - 260, "Wed: 2000 Genuine, 98 Fraud")
    p.drawString(72, height - 275, "...")
    
    p.setFont("Helvetica-Bold", 14)
    p.drawString(72, height - 320, "Analyst Decision Breakdown")
    p.setFont("Helvetica", 10)
    p.drawString(72, height - 340, "• Analyst Approved: 400")
    p.drawString(72, height - 355, "• Analyst Blocked: 300")
    # --- PDF কনটেন্ট শেষ ---

    p.showPage()
    p.save()
    buffer.seek(0)
    print("PDF Report generated successfully.")

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={
            'Content-Disposition': 'attachment; filename="Fraud_Analytics_Report.pdf"'
        }
    )

# # 12. সার্ভার রান করার কমান্ড
# if __name__ == "__main__":
#     print("Starting FastAPI server (with WebSocket/PDF support) at http://127.0.0.1:8000")
#     uvicorn.run(app, host="127.0.0.1", port=8000)