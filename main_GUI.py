from tkinter import *
from tkinter import ttk
from threading import Thread
import tkinter.font as tkfont
import tobii_research as tr
from PIL import Image, ImageTk
import numpy as np
import math
import os
import json
import base64
from io import BytesIO
from datetime import datetime



'''INITIALIZED GLOBAL VARIABLES: each used in one or more functions below'''

#PIXEL DIMENSIONS USED TO CALCULATE CORRESPONDING LOCATION ON IMAGE FOR GIVEN GAZE COORDINATE
widget_width,widget_height=0,0
left_x, top_y = 0,0

#INFORMATION ABOUT IMAGE TO HELP DATA ANALYSIS/VISUALIZATION
#NOTE: replace folder_path with whatever path contains desired images to be shown in experiment
image_number = None
image_id = None
resized_image = None
folder_path = "C:/Users/ljdde/Downloads/CVC/image_samples"
file_names=[filename for filename in os.listdir(folder_path)]

#BOOLEANS TO INDICATE PROGRESS ALONG DIFFERENT POINTS OF EXPERIMENT (collecting_data)
collecting_data=False
done_collecting=False
experiment_done=False
first_time=True

#OCR DATA -- contains data on each of the images of DocVQA dataset Task 1
ocr_data = np.load("C:/Users/ljdde/Downloads/CVC/spdocvqa_imdb/imdb_train.npy",allow_pickle=True)

#DateTime to use as marker to organize data collection
now = datetime.now()
formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")

current_trial_data = {formatted_now:{}}


#RETRIEVE ALL EXPERIMENT DATA
current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, 'experiment_data.json')

with open(json_path, 'r') as json_file:
    experiment_data = json.load(json_file)




def save_data():
    experiment_data[formatted_now]=current_trial_data[formatted_now]
    with open(json_path,'w') as json_file:
        json.dump(experiment_data, json_file, indent=4)

def get_file_question(id):
    '''
    Given an Image ID, returns the question associated with the image

    Parameters:
    id (str): Image ID. Found within the image_next() function

    Returns:
    str: string that is the question corresponding to given image (in the context of DocVQA dataset Task 1)
    '''
    
    for file in ocr_data[1:]:
        if file['image_id']==id:
            return file['question']
    
    raise(Exception(f'Image ID {id} not found'))

def get_file_number(id):
    '''
    Given an Image ID, returns associated index of that image in the DocVQA dataset
    
    Parameters:
    id (str): Image ID. Found within the image_next() function

    Returns:
    int: integer that is used to index into DocVQA dataset to access the given image's data information
    '''
 
    for index, file in enumerate(ocr_data[1:]):
        if file['image_id']==id:
            return index+1
        

def fit(filename,reduced_prop=.8):
    '''
    Returns a new image that is at most reduced_prop size of the screen in both height and width
    
    Parameters:
    filename (str): the filename for the image
    reduced_prop (float): a float between 0 and 1 that indicates that maximal proportion of the screen 
                            you want the image to take up in both height & width. Default value is .8

    Returns:
    Image: a Pillow Image object that is resized to fit the screen
    '''

    image = Image.open(filename)
    #this block is resizing the image until it is smaller than constraints defined by reduced_prop
    while image.size[0]>reduced_prop*screensize[0] or image.size[1]>reduced_prop*screensize[1]:
        image = image.resize((math.floor(.95*image.size[0]),math.floor(.95*image.size[1])))
    return image

#move to next page function
def nextpage():
    '''
    Changes the tkinter screen to the next IMAGE/QUESTION page. 

    '''
    global collecting_data, first_time

    collecting_data = True #boolean to indicate data collection to begin -- see function run_eye_tracker()

    if first_time: #transition out of Welcome page

        #get rid of home_page widgets
        welcome.grid_remove()
        instr.grid_remove()
        instr_head.grid_remove()
        start.grid_remove()
        centerframe['padding']=0

        #initialize experiment widgets
        question.grid(column=0,row=0,columnspan=2,pady=header_len)
        text_response.grid(column=0,row=2)
        response.grid(column=1,row=2)
        document.grid(column=0,row=1,columnspan=2)

        first_time=False #no longer first time
    else:
        next_question.grid_remove()  #remove Next Question button

        #reset and re-visualize user_response entry
        user_response.set('')
        text_response.grid(column=0,row=2)
        response.grid(column=1,row=2)
        
    

    #call function that handles fetching of next image/question pair
    image_next()
    

def image_next():
    '''
    Handles the actual fetching of the next image/question pair from the given folder.

    
        NOTE: need to still implement fetch random image from sample_images folder
    '''
    global resized_image, image_number, image_id
    global widget_height,widget_width,left_x,top_y, experiment_done


    if len(file_names)!=0:  #NOTE: need to change this so that its dependent on parameter for how many images to be shown 

        new_image_name = file_names.pop(0)
        new_image_path = os.path.join(folder_path,new_image_name)

        
        image_id = new_image_name[:-4] #IMAGE ID -- ONLY valid for 3-letter image files (PNG,JPG,etc) -- otherwise will require modification


        resized_image=fit(new_image_path) #Resize and format image to prepare for visualization in GUI
        formatted_img=ImageTk.PhotoImage(resized_image)


        #add image as widget for next page
        document['image']=formatted_img
        document.image=formatted_img
        document.grid(column=0,row=1,columnspan=2)

        #focus mouse on user entry    
        response.focus_set()
        
        #set document question
        image_number = get_file_number(image_id)
        doc_question.set(ocr_data[image_number]['question'])


        #get widget width and length
        root.update()
        widget_width , widget_height = document.winfo_width(),document.winfo_height()

        #calculate display dimensions, necessary for accurate calibration between 
        # gaze data point and corresponding image location
        left_x = leftframe.winfo_width()+(centerframe.winfo_width()-widget_width)//2
        top_y = question.winfo_height()+2*header_len

    else: #once we have no more images to show for the trial
        #remove everything, set experiment_done Boolean to true and save trial data
        document.grid_remove()
        response.grid_remove()
        text_response.grid_remove()
        experiment_done = True
        doc_question.set('All done! Yay! Press the escape key to quit the app')
        save_data()




def threading(): 
    # if experiment still going
    # initialize new thread that will run eye tracker during a given question/image answer process
    if not experiment_done: 
        t1=Thread(target=run_eye_tracker) 
        t1.start() 
    else: #do nothing if experiment over
        pass
   
def run_eye_tracker():
    '''
    Runs eye tracker during question/image process, terminates when an answer is submitted via <Enter>
    Then stores eye-tracking data, along with image properties, to current_trial_data, which will be dumped into JSON

    '''
    #get list of available eye trackers
    found_eyetrackers = tr.find_all_eyetrackers()
    my_eyetracker = found_eyetrackers[0]

    ### PRINT info about eye tracker if desired
    # print("Address: " + my_eyetracker.address)
    # print("Model: " + my_eyetracker.model)
    # print("Name: " + my_eyetracker.device_name)
    # print("Serial number: " + my_eyetracker.serial_number)

    global collecting_data, done_collecting

    current_question_eye_data={}
    done_collecting=False

    def gaze_data_callback(gaze_data):
        '''
        Function that adds gaze data from eye tracker to global dictionary

        Parameters: 
        gaze_data (dict): a data point from a single gaze point

        Returns:
        None: but modifies global dictionary to include current gaze data point
                in the form of {timestamp: (left_eye_coordinate, right_eye_coordinate)}
        '''
        nonlocal current_question_eye_data
        current_question_eye_data[gaze_data['system_time_stamp']]=(gaze_data['left_gaze_point_on_display_area'],
                                                        gaze_data['right_gaze_point_on_display_area'])
        

    # this block continuously subscribes to the gaze data and inputs it into gaze_data_callback function 
    # until is unsubscribed from (determined by Boolean collecting_data which is determined by status of question/image progress)
    # the gaze data is collected at a rate of 60 Hz, according to Tobii spark website

    my_eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)  
    while collecting_data:  
        pass

    my_eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)




    #locally create new dict to hold all info about current question/answer
    current_question_answer_info={}

    def image_to_base64(image):
        buffered = BytesIO()
        format = image.format if image.format else 'PNG'  # Default to 'PNG' if format is None
        image.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    current_question_answer_info['gaze_data']=current_question_eye_data
    current_question_answer_info['edited_image']=image_to_base64(resized_image)
    current_question_answer_info['widget_width']=widget_width
    current_question_answer_info['widget_height']=widget_height
    current_question_answer_info['left_x']=left_x
    current_question_answer_info['top_y']=top_y
    current_question_answer_info['screensize']=screensize
    current_question_answer_info['image_number']=image_number

    current_trial_data[formatted_now][image_id]=current_question_answer_info

    done_collecting = True


    
    
def question_answered():
    '''
    Displays page following question/image. Serves as a transition for user to prepare for next question/image
    
    '''
    document.grid_remove()
    text_response.grid_remove()
    response.grid_remove()
    

    doc_question.set('Good job, click below for next question/image pair')

    next_question.grid(column=0,row=1,columnspan=2) #display button that will take to next image/question pair




def stop_data_collection():
    '''
    Sets collecting_data to false to indicate end of image/question process
    '''
    global collecting_data
    
    if collecting_data:
        collecting_data  = False


    while not done_collecting: #ensure question/answer data is stored before moving on
        pass


    question_answered()






def quit_app(event):
    '''
    Pressing <esc> allows to quit tkinter app
    '''
    root.destroy()
    
    
if __name__ == "__main__": 
    #root
    root = Tk()
    root.attributes('-fullscreen',True)
    root.title("Document Analysis Experiment")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # 'Enter' and 'esc' key functions
    root.bind('<Return>',lambda event: [stop_data_collection()])
    root.bind('<Escape>', quit_app)

    
    # size of display screen -- needed for later calculations
    screensize = root.winfo_screenwidth(), root.winfo_screenheight()
    

    

    #outermost mainframe
    mainframe=ttk.Frame(root)
    mainframe.grid(column=0,row=0,columnspan=3,rowspan=1,sticky=(N, W, E, S))
    mainframe.rowconfigure(0,weight=1)
    for col in [0,2]:
        mainframe.columnconfigure(col,weight=1)
    
    


    #three column frames
    #centerframe will contain all content
    centerframe = ttk.Frame(mainframe)
    centerframe.grid(column=1, row=0, sticky=(N, W, E, S))
    centerframe['relief']='sunken'
    centerframe['padding']=(5,0)

    leftframe = ttk.Frame(mainframe)
    leftframe.grid(column=0, row=0, sticky=(N, W, E, S))
    leftframe['relief']='groove'


    rightframe = ttk.Frame(mainframe)
    rightframe.grid(column=2, row=0, sticky=(N, W, E, S))
    rightframe['relief']='groove'

    
    

    #creating new fonts and styles
    # also establishing size "header_len" that is proportional to the display screen
    header_font = tkfont.Font(family="Consolas", size=25, weight="bold")
    header_len = header_font.measure("m")
    instr_head_font = tkfont.Font(family="Consolas", size=15, weight="normal",underline=1)
    button_style = ttk.Style()
    button_style.configure('start_style.TButton',foreground='black',background='green',font=('Times New Roman',20))

    #welcome page widgets
    # Welcome header, instructions, & start button
    welcome=ttk.Label(centerframe, text="Welcome to Eye Tracking Experiment!",font=header_font)
    welcome.grid(column=0,row=0,pady=header_len*2)

    instr_head=ttk.Label(centerframe,text='Instructions',justify='left',font=instr_head_font)
    instr_head.grid(column=0,row=1,pady=header_len)     


    instructions='You will be shown a sequence of documents, along with a question at the top of the page corresponding to each image. \n Analyze the document and type your answer to the question in the field that will appear at the bottom of the screen, and then press Enter.'
    instr=ttk.Label(centerframe,text=instructions,justify='center')
    instr.grid(column=0,row=2,sticky=(N,W,E,S),pady=(0,20*header_len))

    
    start=ttk.Button(centerframe,text='Start',style='start_style.TButton',command=lambda: [nextpage(),threading()])
    start.grid(column=0,row=3)


    #experiment widgets
    #question text, document image, user response entry

    doc_question = StringVar()
    question_font = tkfont.Font(family="Consolas", size=15, weight="normal")
    question = ttk.Label(centerframe,textvariable = doc_question,font=question_font,background='lightblue')

    document = ttk.Label(centerframe)

    text_response = ttk.Label(centerframe, text = "Answer:")
    user_response = StringVar()
    response = ttk.Entry(centerframe,textvariable = user_response)

    #intermediate button between question/image pairs

    next_question = ttk.Button(centerframe,style='start_style.TButton',text='Next Question',command=lambda: [nextpage(),threading()])

    
    
    root.mainloop()