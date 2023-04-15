import cv2
import openface
import os
from sklearn import svm
import joblib
from joblib import dump
import numpy as np
import dlib
from colorama import init, Fore, Back, Style
init()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def get_face_features(img):
    faces = detector(img,1)
    if len(faces) > 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = predictor(gray, faces[0])
        face_descriptor = np.asarray(facerec.compute_face_descriptor(img, shape))
        return face_descriptor
    else:
        return None

def load_imagesModel(path,names):
    images = []
    labels = []
    for name in names:
        person_path = os.path.join(path, name)
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image = cv2.imread(image_path)
            images.append(image)
            labels.append(name)
    return images, labels
  
def detetectFace(img):
    faces_rects = detector(img, 1)
    aligned_faces = []
    for face_rect in faces_rects:
        aligned_face = dlib.get_face_chip(img, predictor(img, face_rect))
        aligned_faces.append(aligned_face)
    return aligned_faces 

def list_files(root_dir,ex):
    files_list = []
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            if file_name.endswith(ex):
                files_list.append(os.path.join(root, file_name))
    return files_list

def cl():
    os.system('cls' if os.name == 'nt' else 'clear')
while True:
    cl()
    print("in the name of allah")
    print("1. Extract Face From Image")
    print("2. Extract Face From Video")
    print("3. Make Model")
    print("4. Test Model")
    print("5. Exit")
    
    key = input(Fore.BLUE+Back.YELLOW+"Please Select a Number:"+Fore.WHITE+Back.BLACK+" ")
    if key.isdigit():
        key = int(key)
    else:
        key = 0
    
    if key == 5:
        cl()
        break

    if key == 1:
        a = input('Please Enter a Image Path or Image Directory:')
        b = input('Please Enter Output Faces Directory:')
        
        
        files = []
        if a.endswith(".jpg"):
            files.append(a)
        else:
            files = list_files(a,(".jpg"))
        
        i = 0
        print(Fore.GREEN+"Count: 0"+Fore.WHITE,end='\r')
        for file in files:
            img = cv2.imread(file)
            bbs = detetectFace(img)
            for bb in bbs:
                output_path = os.path.join(b, "Pic" + str(i)+".jpg")
                cv2.imwrite(output_path, bb)
                i = i + 1
                print(Fore.GREEN+"Count: "+str(i)+Fore.WHITE,end='\r')

        print("Count Images: ",i,"\n","Finished, Please presss any key...")    
        _ = input()

    if key == 2:
        a = input('Please Enter a Video Path or Video Directory:')
        b = input('Please Enter Output Faces Directory:')
        
        files = []
        if a.endswith(".mp4"):
            files.append(a)
        else:
            files = list_files(a,(".mp4"))
        
        i = 0
        print(Fore.GREEN+"Count: 0"+Fore.WHITE,end='\r')
        for file in files:    
            cap = cv2.VideoCapture(file)
            n = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                n = n + 1
                bbs = detetectFace(frame)
                for bb in bbs:
                    output_path = os.path.join(b, "Pic" + str(i)+".jpg")
                    cv2.imwrite(output_path, bb)
                    i = i + 1
                    print(Fore.GREEN+"Count: "+str(i)+Fore.WHITE,end='\r')
                break
            cap.release()
        print("Count Images: ",i,"\n","Finished, Please presss any key...")    
        _ = input()
    
    if key == 3:
        a = input('Please Enter a Images Directoty:')
        b = input('Please Enter a Model Name:')
        names = []
        for file in os.listdir(a):
            d = os.path.join(a, file)
            if os.path.isdir(d):
                names.append(file)
        images, labels = load_imagesModel(a,names)
        features = []
        newlabels = []
        i = 0
        ni = 0
        print("Image Ok: ",i," Image Ignor",ni,end='\r')
        for img, lab in zip(images, labels):
            feature = get_face_features(img)
            if feature is not None:
                features.append(feature)
                newlabels.append(lab)
                i = i + 1
            else:
                ni = ni + 1
            print(Fore.GREEN + "Images Ok: ",i,Fore.WHITE+" |"+Fore.RED + "Images Ignor: ",ni,Fore.WHITE+" |"+Fore.WHITE+ "All Images: ",str(i+ni),end='\r')

        X_train = np.asarray(features)
        y_train = np.asarray(newlabels)
        
        
        clf = svm.SVC(kernel='linear', probability=True)
        clf.fit(X_train, y_train)
        
        dump(clf, os.path.join(a, b+'.joblib'))
        print("Model Path\n",os.path.join(a, b+'.joblib'))
        print("Train Finished and model saved, Please presss any key...")    
        _ = input()
    
    if key == 4:
        a = input("Video file(1) or WebCam(2)?")
        cap = None
        if int(a) == 1:
            a = input("Please Enter Video Address:")
            cap = cv2.VideoCapture(a)
        else:
            cap = cv2.VideoCapture(0)
        b = input("Model Address:?")
        classifier = joblib.load(b)
        
        
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            bbs = detector(frame, 1)
            
            for bb in bbs:

                shape = predictor(gray, bb)
                face_descriptor = np.asarray(facerec.compute_face_descriptor(frame, shape))
                
                for label in classifier.predict([face_descriptor]):
                    x = classifier.decision_function([face_descriptor])
                    percent_decision = (1 / (1 + np.exp(-x))) * 100

                    feature_array = np.array(x[0])
                    print(label+"|"+str(percent_decision))
                    cv2.putText(frame, label, (shape.part(0).x, shape.part(0).y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cv2.destroyWindow('Frame')
                break
        cap.release()
        
        print("Please presss any key...")    
        _ = input()
