import cv2

# Cargamos los clasificadores pre-entrenados
clasificador_cara = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
clasificador_ojo = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Inicializamos la captura de video
captura = cv2.VideoCapture(0)

# Verificamos que la webcam esté funcionando correctamente
if not captura.isOpened():
    raise IOError("No se puede abrir la webcam")

while True:
    # Capturamos un frame de video
    ret, imagen = captura.read()

    # Si no se pudo capturar el frame, salimos del bucle
    if not ret:
        break

    # Convertimos la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Detectamos caras en la imagen en escala de grises
    caras = clasificador_cara.detectMultiScale(imagen_gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iteramos sobre cada cara detectada
    for (x, y, w, h) in caras:
        # Dibujamos un rectángulo verde alrededor de la cara
        cv2.rectangle(imagen, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Obtenemos la región de interés (ROI) de la cara en la imagen en escala de grises y en la imagen original
        roi_gris = imagen_gris[y:y+h, x:x+w]
        roi_color = imagen[y:y+h, x:x+w]

        # Detectamos ojos en la región de interés de la cara
        ojos = clasificador_ojo.detectMultiScale(roi_gris)

        # Iteramos sobre cada ojo detectado
        for (ex, ey, ew, eh) in ojos:
            # Dibujamos un rectángulo azul alrededor del ojo
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

    # Mostramos el frame con las detecciones en una ventana llamada 'Video en Streaming'
    cv2.imshow('Video en Streaming', imagen)

    # Esperamos a que se presione la tecla 's' para salir del bucle
    if cv2.waitKey(1) == ord('s'):
        break

# Liberamos los recursos de la cámara y cerramos todas las ventanas abiertas
captura.release()
cv2.destroyAllWindows()
