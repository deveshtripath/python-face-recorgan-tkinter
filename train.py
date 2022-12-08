from tkinter import *
from tkinter import ttk
from PIL import Image,ImageTk
from tkinter import messagebox
import pymysql
import cv2
import numpy as np
import os

class Train:
    def __init__(self,root):
        self.root = root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recogition System")

        
        title_lbl = Label(self.root,text="Train Dataset",font=("times new romen",35,"bold"),bg="white",fg="red")
        title_lbl.place(x=0,y=0,width=1530,height=45)


        
        # background Image
        img_top = Image.open(r"photo\facial_recognition_action.jpg")
        img_top = img_top.resize((1530,325),Image.ANTIALIAS)
        self.photoimg_top = ImageTk.PhotoImage(img_top)

        bg_img = Label(self.root, image=self.photoimg_top)
        bg_img.place(x=0,y=55,width=1530,height=325)


        # Button For train 
        train_btn = Button(self.root,text="Train Data",command=self.train_classifier,width=35,font=("times new romen",34,"bold"),bg="blue",fg="white")
        train_btn.place(x=0,y=380,width=1530,height=60)

        
        # background Image
        img_bottom = Image.open(r"photo\bg.jpg")
        img_bottom = img_bottom.resize((1530,710),Image.ANTIALIAS)
        self.photoimg_bottom = ImageTk.PhotoImage(img_bottom)

        bg_img = Label(self.root, image=self.photoimg_bottom)
        bg_img.place(x=0,y=440,width=1530,height=325)


    def train_classifier(self):
        data_dir = ("Data")
        path = [os.path.join(data_dir,file) for file in os.listdir(data_dir)]

        faces = []
        ids = []

        for image in path:
            img = Image.open(image).convert('L')   # to convert into gray scale
            imageNp = np.array(img,'uint8')
            id = int(os.path.split(image)[1].split('.')[1])
            faces.append(imageNp)
            ids.append(id)
            cv2.imshow("Training",imageNp)
            cv2.waitKey(1) == 13 


        ids = np.array(ids)    
        ################## Train The classifier#####################

        # clf =cv2.face.createLBPHFaceRecognizer()
        clf = cv2.faces.LBPHFaceRecognizer_create()
        clf.train(faces,ids)
        clf.write("classifier.xml")
        cv2.destroyAllWindows()
        messagebox.showinfo("Result","Training datasets completed!!!!")








if __name__ == "__main__":
    root = Tk()
    obj = Train(root) 
    root.mainloop()          