import tensorflow as tf
import keras
import numpy as np

new_model = tf.keras.models.load_model('C:\\Users\\user\\Desktop\\assets')
new_model.summary()

maps = {1:'0',2:'1',3:'2',4:'3',5:'4',6:'5',7:'6',8:'7',9:'8',10:'9',11:'A',12:'B',13:'B',14:'C',15:'D',16:'E',17:'F',18:'G',19:'H',20:'I',21:'J',22:'K',23:'L',24:'M',25:'N',26:'O',27:'P',28:'Q',29:'R',30:'S',31:'T',32:'U',33:'V',34:'W',35:'X',36:'Y',37:'Z',38:'a',39:'b',40:'c',41:'d',42:'e',43:'f',44:'g',45:'h',46:'i',47:'j',48:'k',49:'l',50:'m',51:'n',52:'o',53:'p',54:'q',55:'r',56:'s',57:'t',58:'u',59:'v',60:'w',61:'x',62:'y',63:'z'}


import cv2
#import tensorflow as tf
#import keras
while(True):
    
    vid = cv2.VideoCapture(0)

    a=1
    print("####################################################################")

    while(True):
        a = a+1
        check, frame = vid.read()
        print(frame)

        gr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('video',gr)

        key = cv2.waitKey(1)

        if key == ord('q'):
            break

    print("####################################################################")
    print(frame)
    print(a)
    vid.release()
    cv2.destroyAllWindows





    print(frame)



    frame = tf.keras.utils.normalize(gr, axis = 1)


    frame = cv2.resize(frame, (28,28))

    #frame.reshape((28,28))

    print(frame.shape)

    predic = new_model.predict(frame.reshape(1,28,28))

    import numpy as np

    p = np.argmax(predic)

    print(p)

    print("Predicted image:")
    print(maps[p])
          



    k = input("Enter x to stop")

    if(k=='x'):
        break



