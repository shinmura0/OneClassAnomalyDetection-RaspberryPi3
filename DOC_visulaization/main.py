import cv2
import time
import os
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json
from keras import backend as K

threshold = 2
m_input_size, m_input_size = 96, 96

path = "pictures/"
if not os.path.exists(path):
    os.mkdir(path)

model_path = "model/" 
if os.path.exists(model_path):
    # LOF
    #print("LOF model building...")
    #x_train = np.loadtxt(model_path + "train.csv",delimiter=",")

    #ms = MinMaxScaler()
    #x_train = ms.fit_transform(x_train)

    # fit the LOF model
    #clf = LocalOutlierFactor(n_neighbors=5)
    #clf.fit(x_train)

    # Visual model
    print("Visual Model loading...")
    model_visual = model_from_json(open(model_path + 'model_visual.json').read())
    model_visual.load_weights(model_path + 'weights_visual.h5')
    
    # DOC
    print("DOC Model loading...")
    model_doc = model_from_json(open(model_path + 'model_doc.json').read())
    model_doc.load_weights(model_path + 'weights_doc.h5')
    print("loading finish")
    
else:
    print("Nothing model folder")


def main():
    camera_width =  352
    camera_height = 288
    fps = ""
    message = "Push [p] to take a picture"
    result = "Push [s] to start anomaly detection"
    flag_score = False
    picture_num = 1
    elapsedTime = 0
    score = 0
    score_mean = np.zeros(10)
    mean_NO = 0

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

    time.sleep(1)

    while cap.isOpened():
        t1 = time.time()

        ret, image = cap.read()
        image=image[:,32:320]
        if not ret:
            break

        # take a picture
        if cv2.waitKey(1)&0xFF == ord('p'):
            cv2.imwrite(path+str(picture_num)+".jpg",image)
            picture_num += 1

        # quit or calculate score or take a picture
        key = cv2.waitKey(1)&0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            flag_score = True
        if key == ord("p"):
            cv2.imwrite(path + str(picture_num) + ".jpg", image)
            picture_num += 1

        if flag_score == True:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)## 2019.6.12add
            img = cv2.resize(img, (m_input_size, m_input_size))
            img_ = np.array(img).reshape((1,m_input_size, m_input_size,3))/255
            test = model_doc.predict(img_)
            #test = test.reshape((len(test),-1))
            #test = ms.transform(test)
            #score = -clf._decision_function(test)
            
            #visualization
            re = model_visual.predict(test)
            map_ = np.abs(re-img_).reshape((96,96,3))
            jet = cv2.applyColorMap(np.uint8(255*map_), cv2.COLORMAP_JET)
            #jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
            jet = cv2.addWeighted(jet,0.2,img,0.8,2.2)
            image = cv2.resize(jet, (camera_height, camera_height))
    
        # output score
        if flag_score == False:
            cv2.putText(image, result, (camera_width - 350, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            #score_mean[mean_NO] = score[0]
            mean_NO += 1
            #if mean_NO == len(score_mean):
            #    mean_NO = 0

            #if np.mean(score_mean) > threshold: #red if score is big
            #    cv2.putText(image, "{:.1f} Score".format(np.mean(score_mean)),(camera_width - 230, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            #else: # blue if score is small
            #    cv2.putText(image, "{:.1f} Score".format(np.mean(score_mean)),(camera_width - 230, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

        # message
        cv2.putText(image, message, (camera_width - 285, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, fps, (camera_width - 164, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, ( 255, 0 ,0), 1, cv2.LINE_AA)

        cv2.imshow("Result", image)

        # FPS
        elapsedTime = time.time() - t1
        fps = "{:.0f} FPS".format(1/elapsedTime)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
