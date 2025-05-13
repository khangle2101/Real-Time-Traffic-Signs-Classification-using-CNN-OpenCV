import numpy as np
import cv2
import pickle

# Cấu hình camera
frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.75
font = cv2.FONT_HERSHEY_SIMPLEX

# Mở camera
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# Tải model
pickle_in = open("model_trained_epoch30.p", "rb")
model = pickle.load(pickle_in)

# Các hàm xử lý ảnh
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def getClassName(classNo):
    classes = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h',
        'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h',
        'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h',
        'No passing', 'No passing for vehicles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop',
        'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry',
        'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
        'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
        'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
        'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
        'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
        'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
        'Keep left', 'Roundabout mandatory', 'End of no passing',
        'End of no passing by vehicles over 3.5 metric tons'
    ]
    return classes[classNo]

# Vòng lặp chính: Nhận diện biển báo thời gian thực
while True:
    success, imgOriginal = cap.read()
    if not success:
        print("Không đọc được khung hình từ camera.")
        continue

    img = cv2.resize(imgOriginal, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)

    # Dự đoán
    predictions = model.predict(img)
    classIndex = np.argmax(predictions)
    probabilityValue = np.max(predictions)

    # Hiển thị kết quả
    if probabilityValue > threshold:
        cv2.putText(imgOriginal, f"CLASS: [{classIndex}] {getClassName(classIndex)}",
                    (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, f"PROBABILITY: {round(probabilityValue * 100, 2)}%",
                    (20, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
        # cv2.putText(...) dùng để vẽ dòng chữ lên ảnh.
        # Hiển thị tên lớp mà mô hình đã dự đoán:
        # classIndex: số thứ tự lớp (VD: 0, 1, 2, ...)
        # getClassName(classIndex): tên của lớp tương ứng (VD: "Speed Limiqt 30 km/h")
        # (20, 35): vị trí chữ trên ảnh (tọa độ x=20, y=35)
        # font: kiểu chữ (đã định nghĩa là cv2.FONT_HERSHEY_SIMPLEX)
        # 0.75: kích thước font
        # (0, 0, 255): màu chữ là đỏ (RGB)
        # 2: độ dày nét
        # cv2.LINE_AA: kiểu khử răng cưa (anti-aliasing) giúp mịn nét chữ
        # probabilityValue * 100: chuyển giá trị từ 0.0–1.0 thành phần trăm (VD: 0.89 → 89%)
        # round(..., 2): làm tròn đến 2 chữ số thập phân.
        # (20, 75): tọa độ dòng thứ 2 nằm bên dưới dòng lớp.
    cv2.imshow("Result", imgOriginal)

    # Nhấn Q để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
