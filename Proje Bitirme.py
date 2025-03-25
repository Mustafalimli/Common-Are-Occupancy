import cv2

# Gerekli kütüphaneleri içe aktar, eğer yüklü değilse model devre dışı bırakılır.
try:
    import torch
    from ultralytics import YOLO
except ModuleNotFoundError:
    print("'torch' and 'ultralytics' modules are required but not installed. YOLO-based detection will be disabled.")
    YOLO = None

def process_frame(frame, model):
    """
    Her karede, YOLO modeliyle 'person', 'book' ve 'table' nesnelerini tespit eder.
    Tespit edilen nesneler, belirlenen renklerle (insan: yeşil, kitap: mavi, masa: kırmızı) kare içine alınır ve etiketleri ekrana yazdırılır.
    """
    # Model kullanılabilir değilse, işlem yapılmadan görüntü döndürülür.
    if model is None:
        print("Model is not available!")
        return frame

    # Modeli kullanarak nesne tespiti yap.
    results = model(frame)
    print(f"Number of detections: {len(results[0].boxes)}")

    # Initialize person count
    person_count = 0
    
    # Her bir tespit sonucunu işleme al.
    for result in results:
        for box in result.boxes:
            # Kare koordinatlarını al ve tam sayıya çevir.
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Sınıf etiketini al.
            label = result.names[int(box.cls[0])]
            print(f"Detected: {label}")

            # Add 'chair' to detection list and count persons
            if label.lower() in ['person', 'book', 'table', 'chair']:
                if label.lower() == 'person':
                    color = (0, 255, 0)    # Yeşil: insan
                    person_count += 1
                elif label.lower() == 'book':
                    color = (255, 0, 0)    # Mavi: kitap
                elif label.lower() == 'table':
                    color = (0, 0, 255)    # Kırmızı: masa
                elif label.lower() == 'chair':
                    color = (255, 255, 0)  # Turkuaz: sandalye
                else:
                    color = (255, 255, 255)
                
                # Kare içine alma ve etiketi ekrana yazdırma.
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Add person count to bottom right corner
    height, width = frame.shape[:2]
    count_text = f"People: {person_count}"
    text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = width - text_size[0] - 10
    text_y = height - 20
    cv2.putText(frame, count_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame

def main(video_path):
    # Video dosyasını aç.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return
        
    # YOLO modelini yükle, eğer modül mevcutsa.
    model = YOLO("yolov8n.pt") if YOLO else None
    if model is None:
        print("YOLO model could not be loaded!")
    else:
        print("YOLO model loaded successfully!")

    # Video boyutlarını al
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Pencereyi oluştur ve boyutlandır
    cv2.namedWindow("Kütüphane Masa Takip", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Kütüphane Masa Takip", frame_width, frame_height)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Her kareyi işleme al.
        processed_frame = process_frame(frame, model)
        
        # İşlenmiş kareyi anlık olarak ekranda göster.
        cv2.imshow("Kütüphane Masa Takip", processed_frame)
     
    # Kaynakları serbest bırak ve pencereleri kapat.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = r"C:\Users\Mustafa\Desktop\Yeni klasör\camera.mp4"  # Yolu kontrol edin
    main(video_path)
