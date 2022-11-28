from operator import itemgetter
import cv2
from deepface import DeepFace
import numpy as np
frontalface = cv2.CascadeClassifier(
    'frontalface_alt2.xml')
profileface = cv2.CascadeClassifier(
    'profileface.xml')
marge = 70 
# cadrage de tous objets animés
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    ret1, frameEmoji = cap.read()
    ret2, frameAge = cap.read()
    ret3, framegender= cap.read()
# classification des objets:detection l'objet visage
    width = int(cap.get(3))
    height = int(cap.get(4))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tab_faces = []
    faces = frontalface.detectMultiScale(gray, 1.5, 5)
    for (x, y, w, h) in faces:
        tab_faces.append([x, y, x+w, y+h])
    faces = profileface.detectMultiScale(gray, 1.5, 5)
    for (x, y, w, h) in faces:
        tab_faces.append([x, y, x+w, y+h])
    gray2 = cv2.flip(gray, 1)
    faces = profileface.detectMultiScale(gray2, 1.5, 5)
    for (x, y, w, h) in faces:
        tab_faces.append([width-x, y, width-(x+w), y+h])
    tab_faces = sorted(tab_faces, key=itemgetter(0, 1))
    index = 0
# detection des emotions, age, genre
    for (x, y, w, h) in tab_faces:
        if not index or (x-tab_faces[index-1][0] > marge or y-tab_faces[index-1][1] > marge):
            cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)
            output = DeepFace.analyze(frame[y:y+h-1, x:x+w], actions = ('emotion', 'age', 'gender', 'race') , models = None, enforce_detection = False, detector_backend = 'opencv', prog_bar = True)
            emotion = max(output['emotion'].items(), key=itemgetter(1))[0]
            age = output['age']
            gender = output['gender']
            text = emotion + ' ' + \
                str(age) + ' ' + gender
            emoji = "" + \
                emotion + "Emoji.jpg"
            emoji = cv2.imread(emoji, 1)
            emoji = cv2.resize(emoji, (0,0), fx=0.4 ,fy=0.4)
            wi = emoji.shape[1]
            he = emoji.shape[0]
            frameEmoji[y:y+he, x:x+wi] = emoji
# mettre une etiquette de chaque objet détecter
            frameAge = cv2.putText(frameAge, str(age) + ' ans', (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            framegender= cv2.putText(framegender, gender, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            frameEmoji = cv2.putText(frameEmoji,emotion, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            image = np.zeros(frame.shape, dtype=np.uint8)
            image[:height//2, :width//2] = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            image[height//2:, :width//2] = cv2.resize(frameAge, (0, 0), fx=0.5, fy=0.5)
            image[:height//2, width//2:] = cv2.resize(frameEmoji, (0, 0), fx=0.5, fy=0.5)
            image[height//2:, width//2:] = cv2.resize(framegender, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("Frame", image)
            index += 1
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
