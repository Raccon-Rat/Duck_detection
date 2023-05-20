import cv2
import torch
import numpy as np
import matplotlib.path as mplPath

model = torch.hub.load("ultralytics/yolov5", "yolov5m", pretrained=True)

# Diccionario para almacenar los centroides de los objetos
object_centroids = {}
center_points = []
max_ids = 7

def detect_and_track():
    cap = cv2.VideoCapture("Segundo/patos_lineas.mp4")

    while cap.isOpened():
        status, frame = cap.read()

        if not status:
            break

        # Inferencia
        pred = model(frame)
        df = pred.pandas().xyxy[0]
        df = df[df["confidence"] > 0.00]

        # Lista de IDs disponibles para asignar
        available_ids = list(range(1, max_ids + 1))

        for i in range(df.shape[0]):
            bbox = df.iloc[i][["xmin", "ymin", "xmax", "ymax"]].values.astype(int)
            centroid = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

            # Buscar el ID más cercano al objeto actual
            object_id = find_nearest_id(centroid)

            if object_id is None:
                # Asignar un nuevo ID disponible si no hay ningún ID cercano
                if available_ids:
                    object_id = available_ids.pop(0)
                else:
                    # Si no hay IDs disponibles, omitir el objeto
                    continue

            # Actualizar el diccionario de centroides con el nuevo valor
            object_centroids[object_id] = (centroid, cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Dibujar el rectángulo y el ID del objeto en el frame
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(frame,
                        f"ID: {object_id}",
                        (bbox[0], bbox[1] - 15),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255, 255, 255),
                        2)
            
        for pt in center_points:
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def find_nearest_id(centroid):
    min_distance = float('inf')
    nearest_id = None

    for object_id, (prev_centroid, prev_frame_num) in object_centroids.items():
        # Calcular la distancia entre el centroide actual y el centroide previo
        distance = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
        if distance < min_distance:
            min_distance = distance
            nearest_id = object_id

    return nearest_id

def calculate_displacement():
    # Calcular el desplazamiento para cada objeto rastreado
    displacements = {}
    prev_frame_num = 0

    for object_id, (centroid, frame_num) in object_centroids.items():
        if object_id not in displacements:
            displacements[object_id] = 0

        displacement = np.linalg.norm(np.array(centroid) - np.array(object_centroids[object_id][0]))
        frame_difference = frame_num - prev_frame_num
        if frame_difference > 0:
            # Calcular el desplazamiento por frame
            displacement_per_frame = displacement / frame_difference
            # Calcular el desplazamiento total
            total_displacement = displacement_per_frame * frame_difference
            displacements[object_id] += total_displacement

        prev_frame_num = frame_num

    # Imprimir los desplazamientos de los objetos
    for object_id, displacement in displacements.items():
        print(f"Objeto {object_id}: Desplazamiento total = {displacement} píxeles")

if __name__ == "__main__":
    detect_and_track()
    calculate_displacement()