import cv2
import torch
import numpy as np
import matplotlib.path as mplPath

model = torch.hub.load("ultralytics/yolo5", "yolov5s", pretrained = True)

def detector():

    cap = cv2.VideoCapture("Segundo/patos.mp4")

    while cap.isOpened():
        status, frame = cap.read()

        if not status:
            break

        # Inferencia
        pred = model(frame)
        # xmin, ymin, xmax, ymax
        df = pred.pandas().xyxy[0]
        #print(df)
        #print(len(df))
        df = df[df["confidence"] > 0.5]

        for i in range(df.shape[0]):
            bbox = df.iloc[i][["xmin", "ymin", "xmax", "ymax"]].values.astype(int)

            # frame -> (xmin, ymin), (xmax. ymax)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

        cv2.imshow("frame", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()

if __name__ == '__main__':
    
    detector()