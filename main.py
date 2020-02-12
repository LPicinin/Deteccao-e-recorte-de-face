import cv2
import sys
import os

dirPai = "images"
dirResult = "result"
CASCADE_PATH = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


dpai, aux0, aux1, aux2, aux3 = os.walk(dirPai)
subDir = dpai[1]
imgMarcadas = []

for i in range(len(subDir)):
    #carrega todas as imagens da pasta em um vetor
    images = load_images_from_folder(dirPai+"/"+str(subDir[i]))
    for y in range(len(images)):
        imagem_cinza = cv2.cvtColor(images[y], cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(imagem_cinza, 1.1, 3)
        '''if(faces is not None):
            for (x, y, w, h) in faces:
                cv2.rectangle(imagem_cinza, (x, y), (x + w, y + h), (255, 0, 0), 2)'''
        facecnt = len(faces)
        print("Faces Detectadas: %d" % facecnt)
        i = 0
        height, width = imagem_cinza.shape[:2]

        for (x, y, w, h) in faces:
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            faceimg = imagem_cinza[ny:ny + nr, nx:nx + nr]
            lastimg = cv2.resize(faceimg, (64, 64))
            imgMarcadas.append(lastimg);
            i += 1
for i in range(len(imgMarcadas)):
    cv2.imwrite(dirResult + "/image%d.jpg" % i, imgMarcadas[i])

print(str(len(subDir)))
