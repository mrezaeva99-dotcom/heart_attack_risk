import os           
import pickle       
import pandas as pd 
import numpy as np  
import io           
import uvicorn     

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from catboost import CatBoostClassifier
from contextlib import asynccontextmanager

# ==================== 2. КЛАСС МОДЕЛИ (ООП) ====================
class HeartAttackPredictor:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.threshold = 0.5
        
        self.paths = {
            "model": os.path.join(model_dir, "final_model.cbm"),
            "scaler": os.path.join(model_dir, "scaler_fixed.pkl"),
            "features": os.path.join(model_dir, "feature_names.pkl"),
            "threshold": os.path.join(model_dir, "threshold.pkl")
        }

    def load_assets(self):
        if not os.path.exists(self.paths["model"]):
            raise FileNotFoundError(f"Файл не найден: {self.paths['model']}")
        
        self.model = CatBoostClassifier()
        self.model.load_model(self.paths["model"])
        
        with open(self.paths["scaler"], 'rb') as f:
            self.scaler = pickle.load(f)
            
        with open(self.paths["features"], 'rb') as f:
            self.feature_names = pickle.load(f)
            
        #if os.path.exists(self.paths["threshold"]):
        #    with open(self.paths["threshold"], 'rb') as f:
        #        self.threshold = pickle.load(f)
        print("✅ Ассеты загружены")

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        
        # Очистка
        to_drop = ['Heart Attack Risk (Binary)', 'Unnamed: 0', 'id']
        data = data.drop(columns=[c for c in to_drop if c in data.columns], errors='ignore')
        
        # Пол 
        if 'Gender' in data.columns:
            g_map = {'Male': 0, '1.0': 0, '1': 0, 'Female': 1, '0.0': 1, '0': 1}
            data['Gender_encoded'] = data['Gender'].astype(str).str.strip().map(g_map).fillna(0).astype(int)
            data = data.drop(columns=['Gender'])

        # Масштабирование
        if self.scaler:
            scale_cols = [c for c in data.columns if c in self.scaler.feature_names_in_]
            data[scale_cols] = self.scaler.transform(data[scale_cols])

        # Синхронизация колонок
        final_df = pd.DataFrame(index=data.index)
        for col in self.feature_names:
            final_df[col] = data[col] if col in data.columns else 0
                
        return final_df

    def predict(self, df: pd.DataFrame):
        ids = df['id'].values
        processed_data = self._preprocess(df)
        
        # Получаем вероятности
        probs = self.model.predict_proba(processed_data)[:, 1]
        
        print(f"\n--- DEBUG INFO ---")
        print(f"Вероятности (первые 5): {probs[:5]}")
        print(f"Средняя вероятность: {probs.mean():.4f}")
        print(f"Порог: {self.threshold}")
        print(f"------------------\n")
        
        preds = (probs >= self.threshold).astype(int)
        return ids, preds
    
# ==================== 3. ИНИЦИАЛИЗАЦИЯ И LIFESPAN ====================

# 1. Путь к папке
MODEL_DIR = "C:/Users/Rodion/Desktop/heart_attack_risk/models/"

# 2. Создаем экземпляр нашего классного предиктора
predictor = HeartAttackPredictor(MODEL_DIR)

# 3. Механизм управления жизненным циклом приложения
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        predictor.load_assets()
    except Exception as e:
        print(f"Критическая ошибка при старте: {e}")
    yield

# 4. Создаем само приложение
app = FastAPI(
    title="Heart Attack Predictor API",
    description="Проект M1: Предсказание риска сердечного приступа (ООП версия)",
    lifespan=lifespan
)

# ==================== 4. HTML ИНТЕРФЕЙС ====================

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Heart Attack Risk Predictor</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #f4f7f6; display: flex; justify-content: center; padding: 40px; }
        .card { background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); width: 100%; max-width: 600px; }
        h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
        .upload-area { border: 2px dashed #3498db; padding: 40px; text-align: center; border-radius: 10px; cursor: pointer; margin-bottom: 20px; transition: 0.3s; }
        .upload-area:hover { background: #ebf5fb; }
        .btn-group { display: flex; gap: 10px; flex-direction: column; }
        .btn { padding: 12px; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; font-size: 16px; transition: 0.3s; }
        .btn-primary { background: #3498db; color: white; }
        .btn-success { background: #27ae60; color: white; }
        .btn:disabled { background: #bdc3c7; cursor: not-allowed; }
        #status { margin-top: 15px; text-align: center; font-weight: bold; }
        #resultsTable { width: 100%; border-collapse: collapse; margin-top: 20px; display: none; }
        th, td { border: 1px solid #ddd; padding: 10px; text-align: center; }
        th { background: #f8f9fa; }
    </style>
</head>
<body>
    <div class="card">
        <h1>❤️ Heart Attack Risk Predictor</h1>
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <b id="fileName">Выберите CSV файл</b>
            <input type="file" id="fileInput" hidden accept=".csv">
        </div>
        
        <div class="btn-group">
            <button class="btn btn-primary" id="viewBtn" disabled>Посмотреть результаты</button>
            <button class="btn btn-success" id="downloadBtn" disabled>Скачать CSV результат</button>
        </div>
        <div id="status"></div>

        <table id="resultsTable">
            <thead><tr><th>ID</th><th>Предсказание</th></tr></thead>
            <tbody id="tableBody"></tbody>
        </table>
    </div>

    <script>
        const fileInput = document.getElementById('fileInput');
        const viewBtn = document.getElementById('viewBtn');
        const downloadBtn = document.getElementById('downloadBtn');
        const status = document.getElementById('status');

        fileInput.onchange = () => {
            if(fileInput.files[0]) {
                document.getElementById('fileName').innerText = fileInput.files[0].name;
                viewBtn.disabled = false;
                downloadBtn.disabled = false;
            }
        };

        async function handlePredict(endpoint, isDownload) {
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            status.innerText = "Обработка...";
            
            try {
                const response = await fetch(endpoint, { method: 'POST', body: formData });
                if (!response.ok) throw new Error("Ошибка сервера");

                if (isDownload) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url; a.download = 'results.csv'; a.click();
                    status.innerText = "Файл скачан!";
                } else {
                    const data = await response.json();
                    renderTable(data.predictions.slice(0, 10));
                    status.innerText = `Готово! Обработано: ${data.total_patients}`;
                }
            } catch (e) { status.innerText = "Ошибка: " + e.message; }
        }

        function renderTable(preds) {
            const body = document.getElementById('tableBody');
            body.innerHTML = preds.map(p => `<tr><td>${p.id}</td><td>${p.prediction}</td></tr>`).join('');
            document.getElementById('resultsTable').style.display = 'table';
        }

        viewBtn.onclick = () => handlePredict('/predict', false);
        downloadBtn.onclick = () => handlePredict('/download_csv', true);
    </script>
</body>
</html>
"""

# ==================== 5. ЭНДПОИНТЫ И ЗАПУСК ====================

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    """Отображает веб-интерфейс"""
    return HTML_CONTENT

@app.post("/predict")
async def predict_json(file: UploadFile = File(...)):
    try:
        content = await file.read()
        try:
            decoded_content = content.decode('utf-8-sig')
        except UnicodeDecodeError:
            decoded_content = content.decode('cp1251')
            
        df = pd.read_csv(io.StringIO(decoded_content), sep=None, engine='python')
        
        print(f"DEBUG: Получены колонки: {df.columns.tolist()}") 
        
        ids, preds = predictor.predict(df)
        
        return {
            "total_patients": len(preds),
            "predictions": [
                {"id": int(ids[i]), "prediction": int(preds[i])} 
                for i in range(len(preds))
            ]
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc()) 
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/download_csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        try:
            decoded = content.decode('utf-8-sig')
        except:
            decoded = content.decode('cp1251')

        df = pd.read_csv(io.StringIO(decoded), sep=None, engine='python')
        
        # Получаем ID и предсказания из нашего класса
        ids, preds = predictor.predict(df)
        
        # Создаем DataFrame для выгрузки
        result_df = pd.DataFrame({
            "id": ids,
            "prediction": preds
        })
        
        print(f"DEBUG: Генерирую CSV. Строк: {len(result_df)}")

        # Записываем в поток
        stream = io.StringIO()
        result_df.to_csv(stream, index=False)
        
        response = StreamingResponse(
            iter([stream.getvalue()]),
            media_type="test/csv"
        )
        response.headers["Content-Disposition"] = "attachment; filename=results.csv"
        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Блок запуска приложения
if __name__ == "__main__":
    print("Запуск сервера на http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")