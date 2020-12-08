from tkinter import *
import tkinter as tk
root = tk.Tk()
root.resizable(False,False)

MainFrame = tk.Frame(root, width=252, height=252, relief='raised', borderwidth=5)
Frame1 = tk.Frame(MainFrame, width=252, height=40, relief='raised', borderwidth=2, bg = "white")
Frame2 = tk.Frame(MainFrame, width=252, height=82, relief='raised', borderwidth=2, bg = "white")
Frame3 = tk.Frame(MainFrame, width=252, height=82, relief='raised', borderwidth=2, bg = "white")
Frame4 = tk.Frame(MainFrame, width=252, height=82, relief='raised', borderwidth=2, bg = "white")


label = tk.Label(Frame1, text='Welcome to Jane class assistant.\nJust give commands to take notes during \nstudying and classes!')
label.config(font=("Algerian", 8), bg = 'white', fg = '#038376')

strat_button = tk.Button(Frame4, text='Click to get Jane started!')

canvas = tk.Canvas(Frame2,bg = 'white')
#canvas.pack(fill = BOTH, expand = False)
sound = tk.PhotoImage(file = 'C:\\Users\\user\\Desktop\\Jane\\final_image.gif')

canvas.create_image(120,40, image = sound)
x=0
while(x<=252):
    
    if(x>=100 and x<=140):
        pass
    else:
        canvas.create_line(x,0,x,82,dash = (4,2), width = 2)
        
    x = x+10


for frame in [MainFrame, Frame1, Frame2, Frame3,Frame4]:
    frame.pack(expand=True)
    frame.pack_propagate(0)

for widget in [label, canvas,strat_button]:
    widget.pack(expand=True, fill='x', anchor='s')
root.mainloop()

