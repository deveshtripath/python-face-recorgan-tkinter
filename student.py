from tkinter import *
from tkinter import ttk
from PIL import Image,ImageTk
from tkinter import messagebox
import pymysql
import cv2

class Student:
    def __init__(self,root):
        self.root = root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recogition System")

        ############variable############

        self.var_dep = StringVar()
        self.var_course = StringVar()
        self.var_year = StringVar()
        self.var_semester = StringVar()
        self.var_std_id = StringVar()
        self.var_std_name = StringVar()
        self.var_div= StringVar()
        self.var_roll = StringVar()
        self.var_gender = StringVar()
        self.var_dob = StringVar()
        self.var_email = StringVar()
        self.var_phone = StringVar()
        self.var_address = StringVar()
        self.var_teacher = StringVar()


        # first Image
        img = Image.open(r"D:\Mysql\advanceFaceRegocgnigationPython\photo\student.png")
        img = img.resize((500,130),Image.ANTIALIAS)
        self.photoimg = ImageTk.PhotoImage(img)

        f_lbl = Label(self.root, image=self.photoimg)
        f_lbl.place(x=0,y=0,width=500,height=130)
        
        # Second image
        img1 = Image.open(r"D:\Mysql\advanceFaceRegocgnigationPython\photo\facial-recognition-system-concept-260nw-680761540.webp")
        img1 = img1.resize((500,130),Image.ANTIALIAS)
        self.photoimg1 = ImageTk.PhotoImage(img1)

        f_lbl1 = Label(self.root, image=self.photoimg1)
        f_lbl1.place(x=500,y=0,width=500,height=130)
        
        # third Image
        img2 = Image.open(r"D:\Mysql\advanceFaceRegocgnigationPython\photo\student.png")
        img2 = img2.resize((550,130),Image.ANTIALIAS)
        self.photoimg2 = ImageTk.PhotoImage(img2)

        f_lbl2 = Label(self.root, image=self.photoimg2)
        f_lbl2.place(x=1000,y=0,width=550,height=130)


        # background Image
        img3 = Image.open(r"D:\Mysql\advanceFaceRegocgnigationPython\photo\bg.jpg")
        img3 = img3.resize((1530,710),Image.ANTIALIAS)
        self.photoimg3 = ImageTk.PhotoImage(img3)

        bg_img = Label(self.root, image=self.photoimg3)
        bg_img.place(x=0,y=130,width=1530,height=710)

        # Label 
        title_lbl = Label(bg_img,text="Student Management System",font=("times new romen",35,"bold"),bg="white",fg="darkgreen")
        title_lbl.place(x=0,y=0,width=1530,height=45)

        # main frame
        main_frame = Frame(bg_img,bd=2,bg="white")
        main_frame.place(x=10,y=55,width=1500,height=600)

        #left side label frame
        Left_frame = LabelFrame(main_frame,bd=2,bg="white",relief=RIDGE,text="Stubent Detail",font=("times new romen",12,"bold"),fg="red")
        Left_frame.place(x=10,y=10,width=740,height=580)

        
        # image
        img_left = Image.open(r"D:\Mysql\advanceFaceRegocgnigationPython\photo\Adobe.png")
        img_left = img_left.resize((720,130),Image.ANTIALIAS)
        self.photoimg_left = ImageTk.PhotoImage(img_left)

        f_lbl = Label(Left_frame, image=self.photoimg_left)
        f_lbl.place(x=5,y=0,width=720,height=130)

        #currnt course 
        current_course_frame = LabelFrame(Left_frame,bd=2,bg="white",relief=RIDGE,text="Current Course Information",font=("times new romen",12,"bold"),fg="red")
        current_course_frame.place(x=5,y=135,width=720,height=120)


        # departemrnt label
        dep_label = Label(current_course_frame,text="Department",font=("times new romen",12,"bold"),bg="white")
        dep_label.grid(row=0,column=0,padx=10)

        dep_combo = ttk.Combobox(current_course_frame,textvariable=self.var_dep,font=("times new romen",12,"bold"),width=17,state="readonly")
        dep_combo["values"] = ("Select Departement","Computer","IT","Civil","Mechanical")
        dep_combo.current(0)
        dep_combo.grid(row=0,column=1,padx=2,pady=10)

        
        # course label
        course_label = Label(current_course_frame,text="Course",font=("times new romen",12,"bold"),bg="white")
        course_label.grid(row=0,column=2,padx=10,sticky=W)

        course_combo = ttk.Combobox(current_course_frame,textvariable=self.var_course,font=("times new romen",12,"bold"),width=17,state="readonly")
        course_combo["values"] = ("Select course","FE","SE","TE","BE")
        course_combo.current(0)
        course_combo.grid(row=0,column=3,padx=2,pady=10,sticky=W)

        
        # year label
        year_label = Label(current_course_frame,text="Year",font=("times new romen",12,"bold"),bg="white")
        year_label.grid(row=1,column=0,padx=10,sticky=W)

        year_combo = ttk.Combobox(current_course_frame,textvariable=self.var_year,font=("times new romen",12,"bold"),width=17,state="readonly")
        year_combo["values"] = ("Select Year","2020","2021","2022","2023")
        year_combo.current(0)
        year_combo.grid(row=1,column=1,padx=2,pady=10,sticky=W)

        
        # semester label
        semester_label = Label(current_course_frame,text="Semester",font=("times new romen",12,"bold"),bg="white")
        semester_label.grid(row=1,column=2,padx=10,sticky=W)

        semester_combo = ttk.Combobox(current_course_frame,textvariable=self.var_semester,font=("times new romen",12,"bold"),width=17,state="readonly")
        semester_combo["values"] = ("Select semester","Sem-1","sem-2")
        semester_combo.current(0)
        semester_combo.grid(row=1,column=3,padx=2,pady=10,sticky=W)

        

        #class student course 
        class_student_frame = LabelFrame(Left_frame,bd=2,bg="white",relief=RIDGE,text="Class student Information",font=("times new romen",12,"bold"),fg="red")
        class_student_frame.place(x=5,y=260,width=720,height=290)
        
        #student iD
        studentId_label = Label(class_student_frame,bg="white",text="studentId",font=("times new romen",12,"bold"),fg="red")
        studentId_label.grid(row=0,column=0,padx=10,pady=5,sticky=W)

        studentId_entry = ttk.Entry(class_student_frame,textvariable=self.var_std_id ,width=20,font=("times new romen",12,"bold"))
        studentId_entry.grid(row=0,column=1,padx=10,pady=5,sticky=W)
        
        #student name
        studentName_label = Label(class_student_frame,bg="white",text="student Name",font=("times new romen",12,"bold"),fg="red")
        studentName_label.grid(row=0,column=2,padx=10,pady=5,sticky=W)

        studentName_entry = ttk.Entry(class_student_frame,textvariable=self.var_std_name,width=20,font=("times new romen",12,"bold"))
        studentName_entry.grid(row=0,column=3,padx=10,pady=5,sticky=W)
        
        #student div
        student_div_label = Label(class_student_frame,bg="white",text="Class Division",font=("times new romen",12,"bold"),fg="red")
        student_div_label.grid(row=1,column=0,padx=10,pady=5,sticky=W)
        
        div_combo = ttk.Combobox(class_student_frame,textvariable=self.var_div,font=("times new romen",12,"bold"),width=17,state="readonly")
        div_combo["values"] = ("Select div","A","B","C")
        div_combo.current(0)
        div_combo.grid(row=1,column=1,padx=10,pady=5,sticky=W)

        
        #roll 
        roll_no_label = Label(class_student_frame,bg="white",text="Roll No.",font=("times new romen",12,"bold"),fg="red")
        roll_no_label.grid(row=1,column=2,padx=10,pady=5,sticky=W)

        roll_no_entry = ttk.Entry(class_student_frame,textvariable=self.var_roll,width=20,font=("times new romen",12,"bold"))
        roll_no_entry.grid(row=1,column=3,padx=10,pady=5,sticky=W)
        
        #gender
        gender_label = Label(class_student_frame,bg="white",text="gender",font=("times new romen",12,"bold"),fg="red")
        gender_label.grid(row=2,column=0,padx=10,pady=5,sticky=W)

        # gender_entry = ttk.Entry(class_student_frame,textvariable=self.var_gender,width=20,font=("times new romen",12,"bold"))
        # gender_entry.grid(row=2,column=1,padx=10,pady=5,sticky=W)
        gender_combo = ttk.Combobox(class_student_frame,textvariable=self.var_gender,font=("times new romen",12,"bold"),width=17,state="readonly")
        gender_combo["values"] = ("Select gender","Male","Female","Other")
        gender_combo.current(0)
        gender_combo.grid(row=2,column=1,padx=10,pady=5,sticky=W)


        
        #Dob
        dob_label = Label(class_student_frame,bg="white",text="dob",font=("times new romen",12,"bold"),fg="red")
        dob_label.grid(row=2,column=2,padx=10,pady=5,sticky=W)

        dob_entry = ttk.Entry(class_student_frame,textvariable=self.var_dob,width=20,font=("times new romen",12,"bold"))
        dob_entry.grid(row=2,column=3,padx=10,pady=5,sticky=W)

        
        #email
        email_label = Label(class_student_frame,bg="white",text="email",font=("times new romen",12,"bold"),fg="red")
        email_label.grid(row=3,column=0,padx=10,pady=5,sticky=W)

        email_entry = ttk.Entry(class_student_frame,textvariable=self.var_email,width=20,font=("times new romen",12,"bold"))
        email_entry.grid(row=3,column=1,padx=10,pady=5,sticky=W)

        
        #phone
        phone_label = Label(class_student_frame,bg="white",text="phone No :",font=("times new romen",12,"bold"),fg="red")
        phone_label.grid(row=3,column=2,padx=10,pady=5,sticky=W)

        phone_entry = ttk.Entry(class_student_frame,textvariable=self.var_phone,width=20,font=("times new romen",12,"bold"))
        phone_entry.grid(row=3,column=3,padx=10,pady=5,sticky=W)
        
        #address
        address_label = Label(class_student_frame,bg="white",text="address Name :",font=("times new romen",12,"bold"),fg="red")
        address_label.grid(row=4,column=0,padx=10,pady=5,sticky=W)

        address_entry = ttk.Entry(class_student_frame,textvariable=self.var_address,width=20,font=("times new romen",12,"bold"))
        address_entry.grid(row=4,column=1,padx=10,pady=5,sticky=W)
        
        #teacher name
        teacher_label = Label(class_student_frame,bg="white",text="Teacher Name:",font=("times new romen",12,"bold"),fg="red")
        teacher_label.grid(row=4,column=2,padx=10,pady=5,sticky=W)

        teacher_entry = ttk.Entry(class_student_frame,textvariable=self.var_teacher,width=20,font=("times new romen",12,"bold"))
        teacher_entry.grid(row=4,column=3,padx=10,pady=5,sticky=W)

        # radio button
        self.var_radio1 = StringVar()
        radiobtn1 = ttk.Radiobutton(class_student_frame,text="Take Photo Sample",variable=self.var_radio1,value="Yes")
        radiobtn1.grid(row=6,column=0,padx=10,pady=2,sticky=W)
        # radio button
        radiobtn2 = ttk.Radiobutton(class_student_frame,text="No Photo Sample",variable=self.var_radio1,value="No")
        radiobtn2.grid(row=6,column=1,padx=10,pady=2,sticky=W)

        
        #Button frame
        btn_frame = Frame(class_student_frame,bd=2,bg="white",relief=RIDGE)
        btn_frame.place(x=0,y=200,width=715,height=35)

        save_btn = Button(btn_frame,text="Save",command=self.add_data,width=15,font=("times new romen",12,"bold"),bg="blue",fg="white")
        save_btn.grid(row=0,column=0,padx=10,pady=3,sticky=W)

        upadate_btn = Button(btn_frame,text="upadate",command=self.update,width=15,font=("times new romen",12,"bold"),bg="blue",fg="white")
        upadate_btn.grid(row=0,column=1,padx=10,pady=3,sticky=W)

        delete_btn = Button(btn_frame,text="delete",command=self.delete,width=15,font=("times new romen",12,"bold"),bg="blue",fg="white")
        delete_btn.grid(row=0,column=2,padx=10,pady=3,sticky=W)

        reset_btn = Button(btn_frame,text="reset",command=self.reset,width=15,font=("times new romen",12,"bold"),bg="blue",fg="white")
        reset_btn.grid(row=0,column=3,padx=10,pady=3,sticky=W)

        
        btn_frame1 = Frame(class_student_frame,bd=2,bg="white",relief=RIDGE)
        btn_frame1.place(x=0,y=235,width=715,height=35)
        
        Take_photo_btn = Button(btn_frame1,command=self.generate_dataset,text="Take photo Sample",width=35,font=("times new romen",12,"bold"),bg="blue",fg="white")
        Take_photo_btn.grid(row=0,column=0,padx=10,pady=2,sticky=W)
        
        update_btn = Button(btn_frame1,text="update photo",width=35,font=("times new romen",12,"bold"),bg="blue",fg="white")
        update_btn.grid(row=0,column=3,padx=10,pady=2,sticky=W)

        
        #right side label frame
        Right_frame = LabelFrame(main_frame,bd=2,relief=RIDGE,text="Student Detail Inforamtion",font=("times new romen",12,"bold"),bg="white",fg="red")
        Right_frame.place(x=750,y=10,width=730,height=580)

        # image
        img_right = Image.open(r"D:\Mysql\advanceFaceRegocgnigationPython\photo\stu.jpg")
        img_right = img_right.resize((700,130),Image.ANTIALIAS)
        self.photoimg_right = ImageTk.PhotoImage(img_right)

        f_lbl = Label(Right_frame, image=self.photoimg_right)
        f_lbl.place(x=5,y=0,width=720,height=130)


        #==========================Searching System===============================

        #class student course 
        search_frame = LabelFrame(Right_frame,bd=2,bg="white",relief=RIDGE,text="Search System",font=("times new romen",12,"bold"),fg="red")
        search_frame.place(x=5,y=135,width=710,height=70)

        #address
        search_label = Label(search_frame,bg="Red",text="Search By :",font=("times new romen",15,"bold"),fg="white")
        search_label.grid(row=0,column=0,padx=10,pady=5,sticky=W)
        
        search_combo = ttk.Combobox(search_frame,font=("times new romen",12,"bold"),width=14,state="readonly")
        search_combo["values"] = ("Select search","Roll_no","Phone_no")
        search_combo.current(0)
        search_combo.grid(row=0,column=1,padx=2,pady=10,sticky=W)

        
        serach_entry = ttk.Entry(search_frame,width=14,font=("times new romen",12,"bold"))
        serach_entry.grid(row=0,column=2,padx=5,pady=5,sticky=W)
        
        search_btn = Button(search_frame,text="search",width=12,font=("times new romen",12,"bold"),bg="blue",fg="white")
        search_btn.grid(row=0,column=3,pady=2,padx=2,sticky=W)
        
        showAll_btn = Button(search_frame,text="show All",width=12,font=("times new romen",12,"bold"),bg="blue",fg="white")
        showAll_btn.grid(row=0,column=4,pady=2,sticky=W)

                # ==================================table frame========================
        table_frame = Frame(Right_frame,bd=2,bg="white",relief=RIDGE)
        table_frame.place(x=5,y=210,width=710,height=350)

        scroll_x = ttk.Scrollbar(table_frame,orient=HORIZONTAL)
        scroll_y = ttk.Scrollbar(table_frame,orient=VERTICAL)

        self.student_table = ttk.Treeview(table_frame,column=("dep","course","year","Sem","id","name","div","roll","gender","dob","email","phone","address","teacher","photo"),xscrollcommand=scroll_x.set,yscrollcommand=scroll_y.set)
        scroll_x.pack(side=BOTTOM,fill=X)
        scroll_y.pack(side=RIGHT,fill=Y)
        scroll_x.config(command=self.student_table.xview)
        scroll_y.config(command=self.student_table.yview)


        self.student_table.heading("dep",text="Departement")
        self.student_table.heading("course",text="Course")
        self.student_table.heading("year",text="Year")
        self.student_table.heading("Sem",text="Semester")
        self.student_table.heading("id",text="Id")
        self.student_table.heading("name",text="Name")
        self.student_table.heading("roll",text="Roll no.")
        self.student_table.heading("gender",text="Gender")
        self.student_table.heading("div",text="Division")
        self.student_table.heading("dob",text="Date of Birth")
        self.student_table.heading("email",text="Email")
        self.student_table.heading("phone",text="Phone No.")
        self.student_table.heading("address",text="Address")
        self.student_table.heading("teacher",text="Teacher Name")
        self.student_table.heading("photo",text="Photo")
        self.student_table['show']="headings"

        self.student_table.column("dep",width=100)
        self.student_table.column("course",width=100)
        self.student_table.column("year",width=100)
        self.student_table.column("Sem",width=100)
        self.student_table.column("id",width=100)
        self.student_table.column("name",width=100)
        self.student_table.column("roll",width=100)
        self.student_table.column("gender",width=100)
        self.student_table.column("div",width=100)
        self.student_table.column("dob",width=100)
        self.student_table.column("email",width=100)
        self.student_table.column("phone",width=100)
        self.student_table.column("address",width=100)
        self.student_table.column("teacher",width=100)
        self.student_table.column("photo",width=100)

        self.student_table.pack(fill=BOTH,expand=1)
        self.student_table.bind("<ButtonRelease>",self.get_cursor)
        self.fetch_data()

##################################function decration ##################################

    def add_data(self):
        if self.var_dep.get()=="Select Department" or self.var_std_name .get()=="" or self.var_std_id .get()=="":
            messagebox.showerror("Error","All Fields are required",parent=self.root)
        else:
            try:
                conn =pymysql.connect(host="localhost",user="root",password="user",database="face_recognisor")
                cur=conn.cursor()
                cur.execute("insert into student values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",(
                            self.var_dep.get(),
                            self.var_course.get(),
                            self.var_year.get(),
                            self.var_semester.get(),
                            self.var_std_id.get(),
                            self.var_std_name.get(),
                            self.var_div.get(),
                            self.var_roll.get(),
                            self.var_gender.get(),
                            self.var_dob.get(),
                            self.var_email.get(),
                            self.var_phone.get(),
                            self.var_address.get(),
                            self.var_teacher.get(),
                            self.var_radio1.get()
                ))
                conn.commit()
                self.fetch_data()
                self.clear()
                conn.close()
                messagebox.showinfo("Success","Student deatails has been addedd successfully",parent=self.root)
            except Exception as es:
                messagebox.showinfo("Error",f"Due to :{str(es)}",parent=self.root)   

    def fetch_data(self):
        
        con =pymysql.connect(host="localhost",user="root",password="user",database="face_recognisor")
        cur=con.cursor()
        cur.execute("select * from student")
        rows =cur.fetchall()
        if len(rows)!=0:
                self.student_table.delete(*self.student_table.get_children())
                for row in rows:
                        self.student_table.insert('',END,values=row)
                con.commit()        
        con.close()



    def clear(self):
        self.var_dep.set(""),
        self.var_course.set(""),
        self.var_year.set(""),
        self.var_semester.set(""),
        self.var_std_id.set(""),
        self.var_std_name.set(""),
        self.var_div.set(""),
        self.var_roll.set(""),
        self.var_gender.set(""),
        self.var_dob.set(""),
        self.var_email.set(""),
        self.var_phone.set(""),
        self.var_address.set(""),
        self.var_teacher.set(""),
        self.var_radio1.set("")                     


    def get_cursor(self,event=""):

        cursor_row = self.student_table.focus()
        contents=self.student_table.item(cursor_row)
        row=contents['values']
        self.var_dep.set(row[0]),
        self.var_course.set(row[1]),
        self.var_year.set(row[2]),
        self.var_semester.set(row[3]),
        self.var_std_id.set(row[4]),
        self.var_std_name.set(row[5]),
        self.var_div.set(row[6]),
        self.var_roll.set(row[7]),
        self.var_gender.set(row[8]),
        self.var_dob.set(row[9]),
        self.var_email.set(row[10]),
        self.var_phone.set(row[11]),
        self.var_address.set(row[12]),
        self.var_teacher.set(row[13]),
        self.var_radio1.set(row[14]) 



    def update(self):
        if self.var_dep.get()=="Select Department" or self.var_std_name .get()=="" or self.var_std_id .get()=="":
            messagebox.showerror("Error","All Fields are required",parent=self.root)
        else:
            try:
                Update = messagebox.askyesno("Update","Do you want to update student details",parent=self.root)
                if Update>0:
                    con =pymysql.connect(host="localhost",user="root",password="user",database="face_recognisor")
                    cur=con.cursor()
                    cur.execute("update student set Dep=%s,course=%s,Year=%s,Semester=%s,Name=%s,Division=%s,Roll=%s,Gender=%s,Dob=%s,Email=%s,Phone=%s,Address=%s,Teacher=%s,PhotoSample=%s where Student_id=%s ",(
                                                                                                                                self.var_dep.get(),
                                                                                                                                self.var_course.get(),
                                                                                                                                self.var_year.get(),
                                                                                                                                self.var_semester.get(),
                                                                                                                                self.var_std_name.get(),
                                                                                                                                self.var_div.get(),
                                                                                                                                self.var_roll.get(),
                                                                                                                                self.var_gender.get(),
                                                                                                                                self.var_dob.get(),
                                                                                                                                self.var_email.get(),
                                                                                                                                self.var_phone.get(),
                                                                                                                                self.var_address.get(),
                                                                                                                                self.var_teacher.get(),
                                                                                                                                self.var_radio1.get(),
                                                                                                                                self.var_std_id.get()
                                                                                                                ))
                    con.commit()
                    self.fetch_data()
                    self.clear()
                    con.close()
                else:
                    if not Update:
                        return 
                messagebox.showinfo("Success","Student details successfully updated",parent=self.root)
            except Exception as es:
                messagebox.showinfo("Error",f"Due to :{str(es)}",parent=self.root) 


    def delete(self):
        if self.var_std_id.get()=="":
            messagebox.showerror("Error","Id Fields are required",parent=self.root)
        else:
            try:
                delete = messagebox.askyesno("Delete","Do you want to delete student details",parent=self.root)
                if delete>0:
                    con =pymysql.connect(host="localhost",user="root",password="user",database="face_recognisor")
                    cur=con.cursor()
                    cur.execute("delete from student where Student_id=%s",self.var_std_id.get())
                    con.commit()
                    con.close()
                    self.fetch_data()
                    self.clear()
                    messagebox.showinfo("Success","Student has been deleted successfully",parent=self.root)
                else:
                    if not delete:
                        return 
                messagebox.showinfo("Success","Student Deleted successfully",parent=self.root)
            except Exception as es:
                messagebox.showinfo("Error",f"Due to :{str(es)}",parent=self.root) 


    def reset(self):
        self.var_dep.set("Select Departement"),
        self.var_course.set("Select Course"),
        self.var_year.set("Select Year"),
        self.var_semester.set("Select Semester"),
        self.var_std_id.set(""),
        self.var_std_name.set(""),
        self.var_div.set("Select Division"),
        self.var_roll.set(""),
        self.var_gender.set(""),
        self.var_dob.set("Male"),
        self.var_email.set(""),
        self.var_phone.set(""),
        self.var_address.set(""),
        self.var_teacher.set(""),
        self.var_radio1.set("")



    def search_data(self):
            con =pymysql.connect(host="localhost",user="root",password="user",database="face_recognisor")
            cur=con.cursor()
            cur.execute("select * from student where "+ str(self.search_by.get())+" LIKE '%"+str(self.search_txt.get())+"%'")
            rows =cur.fetchall()
            if len(rows)!=0:
                    self.Student_table.delete(*self.Student_table.get_children())
                    for row in rows:
                            self.Student_table.insert('',END,values=row)
                    con.commit()        
            con.close() 

# ===============================Generate Data set===============================


    def generate_dataset(self):
        if self.var_dep.get()=="Select Department" or self.var_std_name .get()=="" or self.var_std_id .get()=="":
            messagebox.showerror("Error","All Fields are required",parent=self.root)
        else:
            try:
                con =pymysql.connect(host="localhost",user="root",password="user",database="face_recognisor")
                cur=con.cursor()
                cur.execute("select * from student")
                myresult=cur.fetchall()
                id=0
                for x in myresult:
                    id+=1
                cur.execute("update student set Dep=%s,course=%s,Year=%s,Semester=%s,Name=%s,Division=%s,Roll=%s,Gender=%s,Dob=%s,Email=%s,Phone=%s,Address=%s,Teacher=%s,PhotoSample=%s where Student_id=%s ",(
                                                                                                                                self.var_dep.get(),
                                                                                                                                self.var_course.get(),
                                                                                                                                self.var_year.get(),
                                                                                                                                self.var_semester.get(),
                                                                                                                                self.var_std_name.get(),
                                                                                                                                self.var_div.get(),
                                                                                                                                self.var_roll.get(),
                                                                                                                                self.var_gender.get(),
                                                                                                                                self.var_dob.get(),
                                                                                                                                self.var_email.get(),
                                                                                                                                self.var_phone.get(),
                                                                                                                                self.var_address.get(),
                                                                                                                                self.var_teacher.get(),
                                                                                                                                self.var_radio1.get(),
                                                                                                                                self.var_std_id.get() == id+1
                                                                                                                ))
                con.commit()
                self.fetch_data()
                self.reset()
                con.close()    

                # ===========Load predifined data====================
                face_classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

                def face_cropped(img):
                    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    faces=face_classifier.detectMultiScale(gray,1.3,5)

                    for (x,y,w,h) in faces:
                        face_cropped = img[y:y+h,x:x+w]
                        return face_cropped 

                cap = cv2.VideoCapture(0) 
                # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

                img_id = 0
                while True:
                    ret, my_frame=cap.read() 
                    if face_cropped(my_frame) is not None:
                        img_id+=1
                        face = cv2.resize(face_cropped(my_frame),(450,450))    
                        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
                        file_name_path = "Data/user."+str(id)+"."+str(img_id)+".jpg"
                        cv2.imwrite(file_name_path,face)
                        cv2.putText(face,str(img_id),(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
                        cv2.imshow("Cropped Face",face)
                    if cv2.waitKey(1) == 13 or int(img_id) == 100:
                        break   
                cap.release()
                cv2.destroyAllWindows()        
                messagebox.showinfo("Result","Gentrating Dataset Completed!!!!")

            except Exception as es:
                messagebox.showinfo("Error",f"Due to :{str(es)}",parent=self.root)




if __name__ == "__main__":
    root = Tk()
    obj = Student(root) 
    root.mainloop()  