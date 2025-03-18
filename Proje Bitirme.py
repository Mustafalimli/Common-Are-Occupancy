import cv2
import torch
from ultralytics import YOLO

def process_frame(frame, model):
    results = model(frame)
    occupied_tables = []
    detected_objects = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            
            detected_objects.append((label, (x1, y1, x2, y2)))
            color = (0, 255, 0) if label in ["chair", "table"] else (255, 0, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    tables = [obj for obj in detected_objects if obj[0] == "table"]
    chairs = [obj for obj in detected_objects if obj[0] == "chair"]
    
    for idx, (label, (tx1, ty1, tx2, ty2)) in enumerate(tables):
        table_full = False
        for _, (cx1, cy1, cx2, cy2) in chairs:
            if cx1 > tx1 and cy1 > ty1 and cx2 < tx2 and cy2 < ty2:
                table_full = True
                break
        
        if table_full:
            occupied_tables.append(idx + 1)
            cv2.putText(frame, f"Masa {idx+1} DOLU", (tx1, ty1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"Masa {idx+1} BOS", (tx1, ty1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame, occupied_tables

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    model = YOLO("yolov8n.pt")  # YOLOv8 modeli yükleniyor

    # Video yazıcıyı başlatıyoruz (çıkış dosyası, codec, fps ve frame boyutları)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID codec
    out = cv2.VideoWriter('output_video.avi', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, occupied_tables = process_frame(frame, model)
        
        # Dolu masalar bilgisini konsola yazdırıyoruz
        print(f"Dolu Masalar: {occupied_tables}")
        
        # İşlenmiş frame'i ekrana gösteriyoruz
        cv2.imshow("Kütüphane Masa Takip", processed_frame)

        # İşlenmiş frame'i çıktı videosuna kaydediyoruz
        out.write(processed_frame)
        
        # 'q' tuşuna basılırsa video kapanacak
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Kaynakları serbest bırakıyoruz
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = r"C:\Users\Mustafa\Desktop\Yeni klasör\camera.mp4"  # Yolu kontrol edin
    main(video_path)
