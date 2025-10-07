from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware # CORS uchun
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import torch.nn as nn # Modelingiz arxitekturasini import qilish kerak
import logging # Loglash uchun

# Loglash sozlamalari
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------ 1. Modelingiz arxitekturasini qayta yuklang ------
# Xuddi treningdagi kabi model klassini bu yerga qo'ying
# Misol uchun, avvalgi SimpleCNN klassi:
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(ImprovedCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) 
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dropout1 = nn.Dropout(0.25) 

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dropout2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dropout3 = nn.Dropout(0.25)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=1)
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(128 * 15 * 15, 512) 
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.3) 
        
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        # x = self.dropout1(x)
        
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        # x = self.dropout2(x)
        
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        # x = self.dropout3(x)
        
        x = self.pool4(x)
        x = self.flatten(x) 
        
        x = self.relu4(self.fc1(x))
        x = self.dropout4(x)
        
        x = self.fc2(x)
        return x

# ------ 2. Modelni yuklash ------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

model_path = 'best_malaria_cnn_model.pth' # Saqlangan modelingiz yo'li
num_classes = 2 # Klasslar soni (masalan, "Parasitized", "Uninfected")
class_names = [ 'Parasitized','Uninfected'] # Klass nomlari

model = ImprovedCNN(num_classes=num_classes)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    logger.info(f"Model loaded successfully from {model_path}")
except FileNotFoundError:
    logger.error(f"Model file not found at {model_path}. Please ensure the path is correct.")
    # Bu yerda dasturni to'xtatish yoki dummy model ishlatish mumkin
    # Hozircha xatolik bilan davom etamiz, lekin productionda boshqacha yondashuv kerak
except Exception as e:
    logger.error(f"Error loading model: {e}")


model.to(device)
model.eval() # Modelni baholash rejimiga o'tkazish juda muhim!

# ------ 3. Rasm uchun transformatsiyalar (treningdagidek) ------
preprocess_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Treningdagi bilan bir xil bo'lishi kerak
])

def transform_image(image_bytes: bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return preprocess_transform(image).unsqueeze(0) # Batch o'lchamini qo'shish (1, C, H, W)
    except Exception as e:
        logger.error(f"Error transforming image: {e}")
        return None

# ------ 4. FastAPI ilovasini yaratish ------
app = FastAPI(title="Malaria Detection API")

# CORS (Cross-Origin Resource Sharing) sozlamalari
# Frontend boshqa domenda/portda ishlayotgan bo'lsa kerak bo'ladi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Barcha manbalarga ruxsat (production uchun aniqroq manbalar ko'rsating)
    allow_credentials=True,
    allow_methods=["*"],  # Barcha metodlarga ruxsat (GET, POST, va hokazo)
    allow_headers=["*"],  # Barcha headerlarga ruxsat
)


@app.get("/")
async def root():
    return {"message": "Malaria Detection API is running. Use the /predict endpoint to make predictions."}

@app.post("/predict")
async def predict_image_endpoint(file: UploadFile = File(...)):
    logger.info(f"Received file: {file.filename}, content type: {file.content_type}")

    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        img_bytes = await file.read() # Faylni asinxron o'qish
        tensor = transform_image(img_bytes)

        if tensor is None:
            logger.error("Image transformation failed.")
            raise HTTPException(status_code=400, detail="Could not transform image. It might be corrupted or in an unsupported format.")

        tensor = tensor.to(device)
        
        with torch.no_grad(): # Gradientlarni hisoblamaslik
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0] 
            _, predicted_idx = torch.max(outputs.data, 1)
        
        predicted_class_name = class_names[predicted_idx.item()]
        confidence_score = probabilities[predicted_idx.item()].item()
        
        logger.info(f"Prediction for {file.filename}: {predicted_class_name} with confidence {confidence_score:.4f}")
        
        return {
            "filename": file.filename,
            "predicted_class": predicted_class_name,
            "confidence": float(confidence_score),
            "probabilities": {class_names[i]: float(probabilities[i].item()) for i in range(len(class_names))}
        }
    except HTTPException as http_exc: # HTTPException ni qayta raise qilish
        raise http_exc
    except Exception as e:
        logger.error(f"Prediction error for {file.filename}: {e}", exc_info=True) # Xatolik haqida to'liq ma'lumot
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during prediction: {str(e)}")

# ------ 5. Uvicorn bilan ishga tushirish ------
# Terminalda quyidagi buyruqni ishlating:
# uvicorn main:app --reload
#
# `main` - Python faylingiz nomi (main.py)
# `app` - FastAPI obyektining nomi (app = FastAPI())
# `--reload` - kod o'zgarganda serverni avtomatik qayta ishga tushiradi (development uchun qulay)