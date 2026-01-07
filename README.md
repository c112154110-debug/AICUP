模型訓練
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo12s.pt')  # 載入預訓練權重
    results = model.train(
        data='aortic_valve.yaml',
        epochs=200,
        imgsz=640,
        batch=8,           # 依顯存調整
        patience=50,
        optimizer='AdamW', # 穩定收斂
        workers=2,         # Windows 避免報錯
        device=0
    )
