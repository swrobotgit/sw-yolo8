import cv2
from ultralytics import YOLO


model = YOLO('yolov8n.pt')
results = model('people.jpg')  # файл изображения
#  документацию читаем тут https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results


for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions

    # Преобразуем изображение в формат OpenCV
    im_array_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)


    # Отображение изображения в окне
    cv2.imshow('Detection Result', im_array)
    cv2.waitKey(0)

    # Сохранение изображения в файл
    cv2.imwrite('results.jpg', im_array)
