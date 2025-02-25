@echo off
echo Installing base dependencies...
python -m pip install --upgrade pip setuptools wheel

echo Installing core requirements...
pip install numpy pandas pillow python-dotenv opencv-python scikit-image

echo Installing OCR dependencies...
pip install pytesseract easyocr
pip install paddlepaddle paddleocr -i https://mirror.baidu.com/pypi/simple
pip install pdf2image

echo Installing ML dependencies...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install scikit-learn transformers

echo Installing visualization dependencies...
pip install streamlit plotly matplotlib

echo Installing utility packages...
pip install pyyaml tqdm pytest pytest-cov

echo Setup complete!
