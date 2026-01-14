import sys
import os
import asyncio
from threading import Thread

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGODB_URL_KEY")
print(mongo_db_url)

import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks
from fastapi.responses import Response, FileResponse, HTMLResponse
from uvicorn import run as app_run
from starlette.responses import RedirectResponse
import pandas as pd

from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

from networksecurity.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from networksecurity.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

# Create necessary directories
os.makedirs("final_model", exist_ok=True)
os.makedirs("prediction_output", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Global variable to track training status
training_status = {
    "is_training": False,
    "status": "idle",
    "message": "No training in progress"
}

@app.get("/", tags=["home"])
async def index(request: Request):
    """Landing page with beautiful UI"""
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        # If template not found, redirect to docs
        return RedirectResponse(url="/docs")

@app.get("/predict-ui", tags=["prediction"])
async def predict_ui(request: Request):
    """Upload page for predictions"""
    try:
        return templates.TemplateResponse("predict.html", {"request": request})
    except Exception as e:
        return HTMLResponse(
            content=f"""
            <html>
                <body style="font-family: Arial; padding: 50px; background: #0d0221; color: white;">
                    <h1>⚠️ Template Not Found</h1>
                    <p>Please ensure the templates folder exists with predict.html</p>
                    <p><a href="/" style="color: #00ffff;">← Back to Home</a></p>
                </body>
            </html>
            """,
            status_code=404
        )

def run_training_pipeline():
    """Background function to run training"""
    global training_status
    try:
        training_status["is_training"] = True
        training_status["status"] = "running"
        training_status["message"] = "Training pipeline started..."
        
        logging.info("Training pipeline initiated via API")
        train_pipeline = TrainingPipeline()
        
        training_status["message"] = "Running training pipeline..."
        train_pipeline.run_pipeline()
        
        training_status["is_training"] = False
        training_status["status"] = "completed"
        training_status["message"] = "Training completed successfully!"
        logging.info("Training pipeline completed successfully")
        
    except Exception as e:
        training_status["is_training"] = False
        training_status["status"] = "failed"
        training_status["message"] = f"Training failed: {str(e)}"
        logging.error(f"Training pipeline failed: {str(e)}")

@app.get("/train", tags=["training"])
async def train_route(background_tasks: BackgroundTasks):
    """
    Endpoint to trigger the training pipeline
    Runs in background to avoid timeout
    """
    global training_status
    
    if training_status["is_training"]:
        return HTMLResponse(
            content=f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Training in Progress</title>
                <meta http-equiv="refresh" content="5">
                <style>
                    body {{
                        font-family: 'Arial', sans-serif;
                        background: linear-gradient(135deg, #0d0221 0%, #1a0b3f 100%);
                        color: white;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        min-height: 100vh;
                        margin: 0;
                    }}
                    .container {{
                        text-align: center;
                        padding: 3rem;
                        background: rgba(255, 255, 255, 0.05);
                        backdrop-filter: blur(20px);
                        border-radius: 24px;
                        border: 1px solid rgba(255, 255, 255, 0.1);
                        max-width: 600px;
                    }}
                    .spinner {{
                        width: 60px;
                        height: 60px;
                        border: 4px solid rgba(0, 255, 255, 0.1);
                        border-top-color: #00ffff;
                        border-radius: 50%;
                        animation: spin 1s linear infinite;
                        margin: 2rem auto;
                    }}
                    @keyframes spin {{
                        0% {{ transform: rotate(0deg); }}
                        100% {{ transform: rotate(360deg); }}
                    }}
                    h1 {{ color: #00ffff; margin-bottom: 1rem; }}
                    .status {{ 
                        color: #ffbe0b; 
                        font-size: 1.2rem; 
                        margin: 2rem 0;
                        animation: pulse 2s ease-in-out infinite;
                    }}
                    @keyframes pulse {{
                        0%, 100% {{ opacity: 1; }}
                        50% {{ opacity: 0.5; }}
                    }}
                    .btn {{
                        display: inline-block;
                        margin-top: 2rem;
                        padding: 1rem 2rem;
                        background: transparent;
                        border: 2px solid #00ffff;
                        color: #00ffff;
                        text-decoration: none;
                        border-radius: 50px;
                        transition: all 0.3s;
                    }}
                    .btn:hover {{
                        background: #00ffff;
                        color: #0d0221;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🤖 Training in Progress</h1>
                    <div class="spinner"></div>
                    <p class="status">{training_status["message"]}</p>
                    <p style="color: rgba(255,255,255,0.6);">
                        This page will auto-refresh every 5 seconds.<br>
                        Training typically takes 2-5 minutes.
                    </p>
                    <a href="/train-status" class="btn">Check Status</a>
                    <a href="/" class="btn">Back to Home</a>
                </div>
            </body>
            </html>
            """,
            status_code=202
        )
    
    # Start training in background
    background_tasks.add_task(run_training_pipeline)
    
    return HTMLResponse(
        content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Training Started</title>
            <meta http-equiv="refresh" content="2;url=/train">
            <style>
                body {
                    font-family: 'Arial', sans-serif;
                    background: linear-gradient(135deg, #0d0221 0%, #1a0b3f 100%);
                    color: white;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    margin: 0;
                }
                .container {
                    text-align: center;
                    padding: 3rem;
                    background: rgba(255, 255, 255, 0.05);
                    backdrop-filter: blur(20px);
                    border-radius: 24px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }
                h1 { color: #00ffff; }
                .icon { font-size: 5rem; margin: 2rem 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="icon">🚀</div>
                <h1>Training Pipeline Started!</h1>
                <p>Redirecting to training status page...</p>
            </div>
        </body>
        </html>
        """,
        status_code=202
    )

@app.get("/train-status", tags=["training"])
async def train_status():
    """Check training status"""
    global training_status
    
    status_color = {
        "idle": "#6c757d",
        "running": "#ffbe0b",
        "completed": "#00ff88",
        "failed": "#ff0055"
    }
    
    return HTMLResponse(
        content=f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Training Status</title>
            {'<meta http-equiv="refresh" content="3">' if training_status["is_training"] else ''}
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    background: linear-gradient(135deg, #0d0221 0%, #1a0b3f 100%);
                    color: white;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    margin: 0;
                }}
                .container {{
                    text-align: center;
                    padding: 3rem;
                    background: rgba(255, 255, 255, 0.05);
                    backdrop-filter: blur(20px);
                    border-radius: 24px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    max-width: 600px;
                }}
                .status-badge {{
                    display: inline-block;
                    padding: 1rem 2rem;
                    background: {status_color[training_status["status"]]};
                    border-radius: 50px;
                    font-weight: bold;
                    text-transform: uppercase;
                    margin: 2rem 0;
                }}
                h1 {{ color: #00ffff; }}
                .message {{
                    font-size: 1.2rem;
                    margin: 2rem 0;
                    color: rgba(255, 255, 255, 0.8);
                }}
                .btn {{
                    display: inline-block;
                    margin: 0.5rem;
                    padding: 1rem 2rem;
                    background: transparent;
                    border: 2px solid #00ffff;
                    color: #00ffff;
                    text-decoration: none;
                    border-radius: 50px;
                    transition: all 0.3s;
                }}
                .btn:hover {{
                    background: #00ffff;
                    color: #0d0221;
                }}
                .spinner {{
                    width: 40px;
                    height: 40px;
                    border: 3px solid rgba(0, 255, 255, 0.2);
                    border-top-color: #00ffff;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                    margin: 1rem auto;
                }}
                @keyframes spin {{
                    0% {{ transform: rotate(0deg); }}
                    100% {{ transform: rotate(360deg); }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Training Status</h1>
                <div class="status-badge">{training_status["status"]}</div>
                {'<div class="spinner"></div>' if training_status["is_training"] else ''}
                <p class="message">{training_status["message"]}</p>
                {'<p style="color: rgba(255,255,255,0.5);">Auto-refreshing every 3 seconds...</p>' if training_status["is_training"] else ''}
                <div>
                    <a href="/" class="btn">🏠 Home</a>
                    <a href="/predict-ui" class="btn">🔍 Make Predictions</a>
                    {'<a href="/train-status" class="btn">🔄 Refresh</a>' if not training_status["is_training"] else ''}
                </div>
            </div>
        </body>
        </html>
        """,
        status_code=200
    )

@app.post("/predict", tags=["prediction"])
async def predict_route(request: Request, file: UploadFile = File(...)):
    """
    Endpoint for making predictions on uploaded CSV file
    """
    try:
        # Read uploaded CSV
        df = pd.read_csv(file.file)
        logging.info(f"Received CSV file with {len(df)} rows and {len(df.columns)} columns")
        
        # Check if model exists
        if not os.path.exists("final_model/preprocessor.pkl") or not os.path.exists("final_model/model.pkl"):
            return HTMLResponse(
                content="""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Model Not Found</title>
                    <style>
                        body {
                            font-family: 'Arial', sans-serif;
                            background: linear-gradient(135deg, #0d0221 0%, #1a0b3f 100%);
                            color: white;
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            min-height: 100vh;
                            margin: 0;
                        }
                        .container {
                            text-align: center;
                            padding: 3rem;
                            background: rgba(255, 255, 255, 0.05);
                            backdrop-filter: blur(20px);
                            border-radius: 24px;
                            border: 1px solid rgba(255, 255, 255, 0.1);
                            max-width: 600px;
                        }
                        h1 { color: #ff0055; }
                        .icon { font-size: 5rem; margin: 2rem 0; }
                        .btn {
                            display: inline-block;
                            margin: 0.5rem;
                            padding: 1rem 2rem;
                            background: linear-gradient(135deg, #00ffff, #ffbe0b);
                            color: #0d0221;
                            text-decoration: none;
                            border-radius: 50px;
                            font-weight: bold;
                            transition: all 0.3s;
                        }
                        .btn:hover {
                            transform: translateY(-3px);
                            box-shadow: 0 10px 30px rgba(0, 255, 255, 0.5);
                        }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="icon">⚠️</div>
                        <h1>Model Not Found</h1>
                        <p>Please train the model first before making predictions.</p>
                        <div style="margin-top: 2rem;">
                            <a href="/train" class="btn">🤖 Train Model</a>
                            <a href="/" class="btn">🏠 Home</a>
                        </div>
                    </div>
                </body>
                </html>
                """,
                status_code=404
            )
        
        # Load model artifacts
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")
        
        # Create NetworkModel instance
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)
        
        # Make predictions
        y_pred = network_model.predict(df)
        logging.info(f"Predictions generated: {y_pred[:10]}")
        
        # Add predictions to dataframe
        df['predicted_column'] = y_pred
        
        # Save to CSV
        output_path = 'prediction_output/output.csv'
        df.to_csv(output_path, index=False)
        logging.info(f"Predictions saved to {output_path}")
        
        # Generate HTML table
        table_html = df.to_html(classes='table', border=0, index=False)
        
        # Return results page
        return templates.TemplateResponse(
            "table.html", 
            {"request": request, "table": table_html}
        )
        
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        return HTMLResponse(
            content=f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Prediction Error</title>
                <style>
                    body {{
                        font-family: 'Arial', sans-serif;
                        background: linear-gradient(135deg, #0d0221 0%, #1a0b3f 100%);
                        color: white;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        min-height: 100vh;
                        margin: 0;
                    }}
                    .container {{
                        text-align: center;
                        padding: 3rem;
                        background: rgba(255, 255, 255, 0.05);
                        backdrop-filter: blur(20px);
                        border-radius: 24px;
                        border: 1px solid rgba(255, 255, 255, 0.1);
                        max-width: 600px;
                    }}
                    h1 {{ color: #ff0055; }}
                    .error {{ 
                        background: rgba(255, 0, 85, 0.1);
                        padding: 1rem;
                        border-radius: 12px;
                        border: 1px solid #ff0055;
                        margin: 2rem 0;
                    }}
                    .btn {{
                        display: inline-block;
                        margin-top: 1rem;
                        padding: 1rem 2rem;
                        background: transparent;
                        border: 2px solid #00ffff;
                        color: #00ffff;
                        text-decoration: none;
                        border-radius: 50px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>⚠️ Prediction Error</h1>
                    <div class="error">
                        <p>{str(e)}</p>
                    </div>
                    <a href="/predict-ui" class="btn">Try Again</a>
                    <a href="/" class="btn">Home</a>
                </div>
            </body>
            </html>
            """,
            status_code=500
        )

@app.get("/download-results", tags=["prediction"])
async def download_results():
    """Download the prediction results as CSV"""
    try:
        output_path = 'prediction_output/output.csv'
        if os.path.exists(output_path):
            return FileResponse(
                output_path,
                media_type='text/csv',
                filename='phishing_detection_results.csv'
            )
        else:
            return Response("No results available. Please run predictions first.", status_code=404)
    except Exception as e:
        logging.error(f"Download failed: {str(e)}")
        raise NetworkSecurityException(e, sys)

@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "NetShield API is running",
        "training_status": training_status["status"],
        "endpoints": {
            "landing": "/",
            "predict_ui": "/predict-ui",
            "predict": "/predict (POST)",
            "train": "/train",
            "train_status": "/train-status",
            "docs": "/docs",
            "download": "/download-results"
        }
    }

if __name__ == "__main__":
    # Fixed: "local host" -> "localhost"
    app_run(app, host="localhost", port=8000)