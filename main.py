import torch
import cv2


#loading the pretrained datasets trained on coco (common objects in contexts - 1.5 milion objects categorised in 80 classes)


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

#opening of the camera as default is 0 and for multiple cameras we can add index as 1,3,2etc
cap = cv2.VideoCapture(0)

selected_object = None

def click_event(event, x, y, flags, param):
    global selected_object
    if event == cv2.EVENT_LBUTTONDOWN:
        for i in results.xyxy[0]:
            xmin, ymin, xmax, ymax, _, class_idx = i
            if xmin < x < xmax and ymin < y < ymax:
                selected_object = model.names[int(class_idx)]
                print(f'Selected Object: {selected_object}')

# create a named window before setting the mouse callback
cv2.namedWindow('Object Recognition and Interaction')

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
     #object detection is being performed
    results = model(frame)
    
    # Drawing  bounding boxes and displaying  labels on it
    for i in results.xyxy[0]:
        xmin, ymin, xmax, ymax, confidence, class_idx = i
        label = model.names[int(class_idx)]
        
       
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {confidence:.2f}', (int(xmin), int(ymin) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    #if an object is selected, display its name on the screen
    if selected_object:
        cv2.putText(frame, f'Selected Object: {selected_object}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # setting the mouse callback function to detect clicks
    cv2.setMouseCallback('Object Recognition and Interaction', click_event)
    
    # Displaying the resulting frame
    cv2.imshow('Object Recognition and Interaction', frame)
    
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
