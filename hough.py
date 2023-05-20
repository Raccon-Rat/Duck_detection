import cv2
import numpy as np

# Función para procesar cada cuadro del video
def process_frame(frame):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar detección de bordes usando el operador Canny
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Aplicar la Transformada de Hough para detectar líneas rectas
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=190)
    
    # Dibujar las líneas detectadas en el cuadro original
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    return frame

# Abrir el archivo de video
video_path = ("Segundo/patos.mp4")
cap = cv2.VideoCapture(video_path)

# Leer el primer cuadro para obtener información del video
ret, frame = cap.read()
height, width, _ = frame.shape

# Crear el objeto de video de salida
output_path = ("Segundo/patos_lineas.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

# Procesar cada cuadro del video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Procesar el cuadro actual
    processed_frame = process_frame(frame)
    
    # Escribir el cuadro procesado en el video de salida
    out.write(processed_frame)
    
    # Mostrar el cuadro procesado en una ventana
    cv2.imshow('Video', processed_frame)
    
    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
out.release()
cv2.destroyAllWindows()