from datetime import date
from tkinter import *
from dateutil import parser
import os


from tkcalendar import *
import tkinter.messagebox as tm

####to save the file

from pathlib import Path

root = Tk() ###initializing tkinter
root.title("My_Thoughts_Hub")
root.geometry("600x600")
#######getting today's date
today = date.today()
y1 = int(today.strftime("%Y"))
m1 = int(today.strftime("%m"))
d1 = int(today.strftime("%d"))

heute = (y1,m1,d1)
#######open the calendar on current date

cal = Calendar(root, selectmode="day", year=y1, month=m1, day=d1)
cal.pack(pady=20, fill= "both", expand=True)

####get date to work on


def get_date2workon():
    tt = parser.parse(cal.get_date(),dayfirst=True)

    y2 = int(tt.strftime("%Y"))
    d2 = int(tt.strftime("%d"))
    m2 = int(tt.strftime("%m"))
    
    return (y2,m2,d2)

####Path to the text file
try:
    os.mkdir("dossier")
except FileExistsError:
    pass
vers = Path('.\dossier')
    
def showcase():
    clicked_date = get_date2workon()

    if (clicked_date > heute):
        ##############save content with date
        textedit()
    elif (clicked_date <heute):

        label1 = Button(master= None, text=" On"+ str(clicked_date) +", You wrote:")
        label1.pack(side= LEFT)
        cal.pack(pady=20, expand=False)
        ######## modify button by output of that day's text
        button2 = Button (root, text= "See text")
        button2.pack(pady=20)
        root.geometry("600x600")
    else:
        ####### Expand the enty input box and save content with date
        textedit()
###read what has been written

def showtext():
    clicked_date = str(lambda:get_date2workon())
    with open (vers, "r") as rt:
        for line in rt:
            if line.startswith(str(clicked_date)):
                print(line)
            else:
                pass #print(line)

###Button to grab the date

button1 = Button (root, text= "Edit", command= showcase)
button2 = Button (root, text= "Read", command= showtext)
button1.pack(pady=20)
button2.pack(pady=20)

def textedit():
    clicked_date = get_date2workon()

    ######## title text input
    label2 = Text(root, height=1, width=50, font="Helvetica 14 bold",)
    label2.pack(pady=20, side= TOP)

    ### create button and avoid it to be clicked upon the execution
    button1.config(text= str(clicked_date)+", Click to save", command= lambda:savetext(label2, label3))
    button1.pack(pady=20, side= BOTTOM)

    ###### body text input
    label3 = Text(master=None, height=100, width=100)
    label3.pack(pady=20, expand=True)
    
    cal.pack(pady=20, expand=False)
    root.geometry("800x800")    

def savetext(title_label, text_label):
    selected_date = get_date2workon()
    title_txt = str(title_label.get(1.0, 'end-1c'))
    typed_txt = str(text_label.get(1.0, 'end'))
    input_text= str(selected_date)+": "+title_txt.upper()+"\n"+ typed_txt

     ### checking if text box is not empty before savin
    if (len(typed_txt.strip()) > 1 ):
        filename= "\{year}-{month}-{day}-{title}.txt".format(year=selected_date[0], month=selected_date[1], day=selected_date[2],title=title_txt)
        path=Path(str(vers)+filename)
        with open (path, 'w+') as dt:
            dt.write(input_text)
        tm.showinfo(title='Saved', message="Text successfully saved")
    else:
        tm.showinfo(title='Empty', message="No text to save")
root.mainloop() ###keeping the GUI open
