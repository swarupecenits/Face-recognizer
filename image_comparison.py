import cv2 as cv
import face_recognition

img = cv.imread("Faces/Messi.webp")
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]

img2 = cv.imread("Faces/Messi.webp")
rgb_img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

result = face_recognition.compare_faces([img_encoding], img_encoding2)
print("Result: ", result)

cv.imshow("Img", img)
cv.imshow("Img 2", img2)
cv.waitKey(0)

