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
        return frame

    # Modeli kullanarak nesne tespiti yap.
    results = model(frame)

    # Her bir tespit sonucunu işleme al.
    for result in results:
        for box in result.boxes:
            # Kare koordinatlarını al ve tam sayıya çevir.
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Sınıf etiketini al.
            label = result.names[int(box.cls[0])]

            # Sadece "person", "book" ve "table" sınıflarına odaklan.
            if label.lower() in ['person', 'book', 'table']:
                # Nesneye göre renk belirle.
                if label.lower() == 'person':
                    color = (0, 255, 0)    # Yeşil: insan
                elif label.lower() == 'book':
                    color = (255, 0, 0)    # Mavi: kitap
                elif label.lower() == 'table':
                    color = (0, 0, 255)    # Kırmızı: masa
                else:
                    color = (255, 255, 255)
                
                # Kare içine alma ve etiketi ekrana yazdırma.
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame

def main(video_path):
    # Video dosyasını aç.
    cap = cv2.VideoCapture(video_path)
    # YOLO modelini yükle, eğer modül mevcutsa.
    model = YOLO("yolov8n.pt") if YOLO else None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Her kareyi işleme al.
        processed_frame = process_frame(frame, model)
        
        # İşlenmiş kareyi anlık olarak ekranda göster.
        cv2.imshow("Kütüphane Masa Takip", processed_frame)
        
        # 1 milisaniyelik bekleme, 'q' tuşuna basılırsa döngüden çık.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Kaynakları serbest bırak ve pencereleri kapat.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "video.mp4"  # Video dosyanın yolu; gerekirse güncelle.
    main(video_path)
