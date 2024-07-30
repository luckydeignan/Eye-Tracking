import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import Button
import hdbscan
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
import random
import json
import base64
from PIL import Image
from io import BytesIO

'''
INITIALIZED HYPERPARAMETERS: edit variables below to your preferences
    More information on recommendations found in their respective 
    function (nearest_ocr() & ocr_cluster()) documentations
'''
fixation_radius = 72 
min_cluster_sample = 5

# loads in data necessary for OCR designation
# NOTE: this is from SQVA Task 1 (https://rrc.cvc.uab.es/?ch=17&com=downloads) training set
#       the current state of code can only analyze images from this dataset 
ocr_data = np.load("C:/path/to/your/OCR/imdb/dataset.npy",allow_pickle=True) #load in ocr data

def gaussian_2d(x, y, x0, y0, A, sigma):
    '''
    Calculates value of (x,y) coordinate following Gaussian_2D distribution from a focal point (x0, y0)
    Used with assumption that gaze_point is not 100% accurate and roughly follows a gaussian distribution outwards
    Solely used for heatmap visualization purposes; not actual data analysis

    Parameters:
        x, y: coordinates of point whose value we desire based on Gaussian distribution
        x0, y0: focal point based on gaze data
        A: value given to focal point
        sigma: standard deviation of distribution

        NOTE: when we use this in the heatpoint() function, we use A=100 and sigma=2 as these 
                were found empirically to depict relatively clean and appealing heatmaps. 
                These values could be played with though

    Returns:
        float: value associated with (x,y) given that A was associated with (x0,y0)
    '''
    return A * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))


def heatpoint(heatmap,x,y):
    '''
    Modifies heatmap to increment (x,y) by A with radially decreasing
    incrementations according to gaussian distribution

    Parameters:
        heatmap (2D np array): the heatmap associated with gaze data
        x&y coordinates (ints): pixel coordinates corresponding to image location that was being gazed upon
    
    Returns: 
        None: modifies the 2D heatmap array to include increased value on given pixel coordinate 
    '''
    A = 100
    sigma = 2
    #all pixels within 3 units (horizontally & laterally) are also incremented according to gaussian distribution
    for y_diff in range(-3,4,1):
        for x_diff in range(-3,4,1):
            heatmap[y+y_diff][x+x_diff] = heatmap[y+y_diff][x+x_diff] + gaussian_2d(x+x_diff,y+y_diff,x,y,A,sigma)

def screen_to_image_pixels(x,y,x_min,x_max,y_min,y_max,screen_width,screen_height):
    '''
    Converts screen proportion coordinates to pixel value coordinates 
    (relative to image -- i.e (0,0) is top-left of image), primarily used in heatmap construction

    Parameters:  
        x,y (floats): screen coordinates in units that are proportion of screen width/height
        x_min,x_max,y_min,y_max (ints): pixel coordinates that bound the document being displayed
        screen_width, screen_height (ints): pixel values of screensize

    Returns: 
        adjusted_x, adjusted_y (ints): pixel value coordinates that correspond to inputted proportional coordinates 
                                    ONLY if coordinates lie on the document being displayed. Else, None is returned.

    '''
    try:
        x = screen_width*x
        y = screen_height*y
        if x_min<=x<=x_max and y_min<=y<=y_max: #ensures pixel coordinates are only returned if they correspond to spot on displayed image
            adjusted_x = x - x_min
            adjusted_y = y - y_min
            return (adjusted_x,adjusted_y)
        else:
            return None, None
    except: # usually occurs when a cluster is located to be off the screen; 
            # likely a result of faulty calibration of the eye tracker
        raise(Exception)
    

    
def image_prop_to_image_pixels(x,y,x_min,x_max,y_min,y_max):
    '''
    Converts image proportion coordinates to image pixel coordinates (relative to image)
    Used to find nearest OCRs in ocr_cluster()

    Parameters:
        x,y: coordinates in units of proportion of image width & height
        x_min,x_max: pixel value that bounds left and right side of image
        y_min,y_max: pixel values that bound top and bottom of image

    Returns:
        relative_x, relative_y: coordinates in units of pixel values relative to image
    '''
    img_width = x_max-x_min
    img_height = y_max-y_min
    relative_x , relative_y = x*img_width , y*img_height
    return relative_x, relative_y

#COLOR FUNCTIONS for cluster visualization purposes

def generate_random_color(threshold=0.2):
    '''Generate a random hexadecimal color that is not too close to white or black.'''

    def rgb_to_hex(r, g, b):
        '''Convert RGB to Hexadecimal.'''
        return f'#{r:02x}{g:02x}{b:02x}'

    def calculate_lightness(r, g, b):
        '''Calculate the lightness of a color given RGB values.'''
        r_prime = r / 255.0
        g_prime = g / 255.0
        b_prime = b / 255.0
        c_max = max(r_prime, g_prime, b_prime)
        c_min = min(r_prime, g_prime, b_prime)
        lightness = (c_max + c_min) / 2.0
        return lightness

    while True:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        lightness = calculate_lightness(r, g, b)
            
        # Ensure the lightness is not too close to black (0) or white (1)
        if threshold < lightness < (1 - threshold):
            return rgb_to_hex(r, g, b)


def nearest_ocr(ocr_info,gaze_location,kdtree,rad=fixation_radius):
    '''
    Parameters: 
    ocr_info (list of dicts): this is gathered from imdb data. Each dict is an OCR in the image, containing its location
    avg_gaze_location (tuple): the avg coordinates of the cluster being inquired. Units are proportion of file
    kdtree (class KDTree): scipy class which allows for efficient nearest neighbors searching given coordinates of data
    rad (float): radial distance, in units of screen proportion, that indicates the region of OCRs to be captured for a given gaze point
    NOTE: due to difference in height/width of display, this will not result in a perfect circle around a gaze point

    Returns: 
    (str): nearest OCR tokens as a string
    '''

    #NOTE: rad is important parameter. Determines the radius around fixation point that catches all the OCRs. 
    # The current default value is 72 pixels. This was calculated based on several assumptions.
    # In perceptual span for reading, width of single fixation is about 18 letter spaces under 12 pt font (14-15 to the right, 3-4 to the left)
    # Assuming character width to be about half of character height (rule of thumb used in estimation, but another assumption)
    # This equates to 108 pt width of fixation point, equating to a radius of 72 pixels
    # The fixation is more akin to an off-centered ellipse though, so still not an entirely accurate model
    # RECOMMENDATION FOR FUTURE: create some tool that draws an elliptical region around a fixation point and use this to catch OCRs

    indeces = kdtree.query_ball_point(gaze_location,rad) #scipy module

    nearest_tokens = [ocr_info[index] for index in indeces]

    return([token['word'] for token in nearest_tokens])



def ocr_cluster(eye_data,image_number,screensize,reduced_width,reduced_height,widget_width,widget_height,left_x,top_y,min_cluster_size=min_cluster_sample):
    '''
    Takes in tracking data and image, clusters data by time and space, and returns information regarding cluster graph and ocr designation.

    Parameters:
        eye_data (dict): eye-tracking data for a given image/question
        image information: already outlined screen_to_image_coords
        min_cluster_size (int): minimum size for data to be considered a cluster. See subsequent "NOTE" in function comments for more info

    Returns:
        x, y, z (lists): time, x-location, y-location of the tracking data
        list_colors (list of strings): list containing the colors for each data point in the data
        cluster_segment_info (list of tuples of floats): list containing information regarding each cluster, used for temporally visualization eye tracking on heatmap using arrows
        ocr_designation (dict): dict containing each cluster and corresponding nearest OCRs within given screen proportion radius
    '''
    x = []
    y = []
    z = []

    #prepare data into a 2D numpy array
    #rows = data point
    #columns = features
    for time in eye_data:
        try:
            x.append(float(time))
            y.append((eye_data[time][0][0]+eye_data[time][1][0])/2) #averaging between right and left eye points
            z.append((eye_data[time][0][1]+eye_data[time][1][1])/2)
        except:
            print(f'Subject looked away from screen at {eye_data[time]}') 

    data = np.column_stack((x,y,z))

    #standardize the data then put into clusters and obtain corresponding labels
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)


    #NOTE: min_cluster_size parameter is important. Default value is 5 as this will catch shortest of fixations
    # Eye-tracking data samples at 60Hz, meaning 1 every 16.67 ms
    # Avg reading fixation duration 200-250 ms, often in 125-175 ms range, can be as low as 75 ms
    # 
    # Given these numbers, min_cluster_size of 5 will catch even the smallest of fixations, but avg cluster size should be ~12-15
    # This logic and implementation can be tested and played around with for further adjustment as needed
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size)


    labels = clusterer.fit_predict(standardized_data) #get arbitrary cluster labels

    
    #prepare number of colors needed to mark all clusters
    colors=[generate_random_color() for _ in range(1+max(labels))]

    #initalize list of len(data_points)
    #each element corresponding to color of that point
    list_colors=[]

    #initialize dict to keep track of average locations for each cluster
    # has format {cluster_pt:[avg_time,avg_x,avg_y]}
    cluster_info={pt:[0,0,0] for pt in range(max(labels)+1)}

    for i, pt in enumerate(labels):
        #if a point doesn't belong to cluster, designate as grey
        if pt<0:
            list_colors.append('grey')
        #otherwise, designate as corresponding cluster color
        else:
            list_colors.append(colors[pt])
            cluster_info[pt][0]+=float(data[i][0])
            cluster_info[pt][1]+=float(data[i][1])
            cluster_info[pt][2]+=float(data[i][2])


    #finalize cluster avg locations
    for pt in cluster_info:
        freq=list(labels).count(pt)
        cluster_info[pt][0]=cluster_info[pt][0]/freq
        cluster_info[pt][1]=cluster_info[pt][1]/freq
        cluster_info[pt][2]=cluster_info[pt][2]/freq
 
    cluster_info=sorted(cluster_info.items(), key= lambda item: item[1][0]) #sort clusters by chronological order // changes cluster_info to list format
    
    cluster_segment_info=[] # initialize list that will contain locations for each cluster appearing on the file // used later for visualization purposes

    #pixel values that bound the document image
    x_max = left_x + widget_width - (widget_width-reduced_width)//2
    y_max = top_y + widget_height - (widget_height-reduced_height)//2
    x_min = left_x + (widget_width-reduced_width)//2
    y_min = top_y + (widget_height-reduced_height)//2


    #this next section prepares the OCR data and builds KD tree
    #this data and tree will be used in nearest OCR calculations

    ocr_info = ocr_data[image_number]['ocr_info'] #ocr info for this specific image

    ocr_designation = {} # going to have each color cluster with corresponding OCRs

    ocr_prep = [] #preparing location data for each ocr

    for ocr in ocr_info: 
        #as of now we are using center of ocr box location to calculate nearest OCRs
        #this could be played with; could potentially use any edge of box to calculate nearest OCRs
        avg_x = ocr['bounding_box']['topLeftX']+(ocr['bounding_box']['width']/2)
        avg_y = ocr['bounding_box']['topLeftY']+(ocr['bounding_box']['height']/2)
        x_pixel, y_pixel = image_prop_to_image_pixels(avg_x,avg_y,x_min,x_max,y_min,y_max)
        ocr_prep.append((x_pixel,y_pixel))

    ocr_locations = np.array(ocr_prep)

    tree = KDTree(ocr_locations) #using KDTree for more efficient searching


    #this next block calculates nearest OCRs for each cluster

    for cluster in cluster_info:
        #convert display proportion coordinates to image proportion coordinates
        final_x, final_y = screen_to_image_pixels(cluster[1][1],cluster[1][2],x_min,x_max,y_min,y_max,screensize[0],screensize[1])
            
        #if cluster lies on screen, find nearest OCR and add to ocr_designation
        if final_x and final_y:
            near_ocr = nearest_ocr(ocr_info,(final_x,final_y),tree)
            cluster_segment_info.append((cluster[1][0],cluster[1][1],cluster[1][2]))
            ocr_designation[colors[cluster[0]]]=near_ocr
        else:
            ocr_designation[colors[cluster[0]]]=None




    #return data to be displayed on cluster graph
    return (x,y,z,list_colors,cluster_segment_info,ocr_designation)


def analyze_data(gaze_data,edited_image,widget_width,widget_height,left_x,top_y,screensize,image_number):
    '''
    Primary function that executes entire analysis of gaze data. Creates and visualizes heatmap corresponding to gaze data 
    overylaying the file image, creates and visualizes clusters which represent fixation points, and returns nearest OCRs within
    a given screen proportion radius of each fixation points. 

    Parameters:
        gaze_data (dict): eye tracking data
        edited_image (Pillow Image): image being gazed upon
        widget_width (int): width of widget in pixels
        widget_height (int): height of widget in pixels
        left_x (int): # of pixels to the left of widget containing the image
        top_y (int): # of pixels above the widget containing the image
        screensize (tuple): size of display being used for experiment
        image_number (int): used to locate image information in dataset

    Visualizes:
        heatmap of eye-tracking gaze data, with arrows at each cluster which indicated the chronological order upon which they were gazed at
        scatterplot showing the actual data of the clusters
        
    Returns:
        ocr_designation (dict): dictionary containing the color of a cluster (HEX format) as keys, and all of its nearby
                                OCRs as its values
    '''

    
    #(height,width) of image and screensize
    height=edited_image.size[1]
    width=edited_image.size[0]
    screen_width, screen_height = screensize[0],screensize[1]

    #initialize array representing image and overlaying heatmap
    image_array = np.array(edited_image)
    heatma=np.zeros((height,width))

    #calculate pixel values that bound the document image
    x_max = left_x + widget_width - (widget_width-width)//2
    y_max = top_y + widget_height - (widget_height-height)//2
    x_min = left_x + (widget_width-width)//2
    y_min = top_y + (widget_height-height)//2

    #add each data point to the heatmap
    for stamp in gaze_data:
        for eye in [0,1]: #to account for both left and right eye data
            try:
                #convert screen proportion to image pixel values and add to heatmap
                adjusted_x,adjusted_y = screen_to_image_pixels(gaze_data[stamp][eye][0],gaze_data[stamp][eye][1],x_min,x_max,y_min,y_max,screen_width,screen_height)
                heatpoint(heatma,round(adjusted_x),round(adjusted_y))
            except:
                #if eye-tracking value == ('nan','nan')
                #this just means user was not looking at screen
                pass
 

    #initialize plots
    fig,ax = plt.subplots(1,2)

    #create first plot; heatmap
    ax[0].imshow(image_array,cmap='gray')
    ax[0].imshow(heatma,cmap='jet',alpha=.5,aspect='auto',vmin=0,vmax=np.max(heatma))
    ax[0].set_title("Heatmap")


    #retrieve data for cluster plot, arrow segments, and ocr designations
    (x,y,z,list_colors,clusters,ocr_designation) = ocr_cluster(gaze_data,image_number,screensize,width,height,widget_width,widget_height,left_x,top_y)

    #this block creates the visual arrow segments that indicate the temporal progress of eye gazing
    segments=[]
    for index, avg_cluster in enumerate(clusters):
        if index==len(clusters)-1:
            continue
        start_x,start_y = screen_to_image_pixels(clusters[index][1],clusters[index][2],x_min,x_max,y_min,y_max,screen_width,screen_height)
        end_x,end_y = screen_to_image_pixels(clusters[index+1][1],clusters[index+1][2],x_min,x_max,y_min,y_max,screen_width,screen_height)
        segments.append([[start_x,start_y],[end_x,end_y]])

    # Add the arrows
    arrows = []
    for seg in segments:
        arrow = FancyArrowPatch((seg[0][0], seg[0][1]),(seg[1][0],seg[1][1]),
                                arrowstyle='->', mutation_scale=15, color='red')
        ax[0].add_patch(arrow)
        arrows.append(arrow)

    # Function to toggle arrow visibility
    def toggle_arrows(event):
        for arrow in arrows:
            arrow.set_visible(not arrow.get_visible())
        plt.draw()

    # Add a button to toggle arrow visibility
    ax_toggle = plt.axes([0.8, 0.01, 0.1, 0.05])
    btn_toggle = Button(ax_toggle, 'Toggle Arrows')
    btn_toggle.on_clicked(toggle_arrows)

    #create second plot; clustered scatterplot
    ax[1]  = fig.add_subplot(122, projection='3d')
    ax[1].scatter(x, y, z, c=list_colors)


    # Set labels for scatterplot
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('x-location')
    ax[1].set_zlabel('y-location (inverted)')

    ### if you want to be able to see OCR designations alongside data visualization
    # for color,ocrs in ocr_designation.items(): 
    #     print(f"{color} cluster was near the following OCRs: {ocrs}")

    plt.show()

    return ocr_designation

    

def base64_to_image(base64_str): 
    '''
    Converts base64 string into Pillow image object
    '''
    # Decode the base64 string to bytes
    img_data = base64.b64decode(base64_str)
    
    # Load the bytes into a BytesIO object
    buffered = BytesIO(img_data)
    
    # Open the image using Pillow
    return Image.open(buffered)

def analyze_single_image(image_id):
    '''
    Performs data analysis on a single image for all experiments that have been conducted
    NOTE: this could result in a long execution time if this image has been collected data from many times
            the code will continuously be interrupted by matplotlib, the script only continues after each plot is closed by the user

    Parameters:
        image_id (str): the string ID that identifies the desired image to be analyzed
    
    Visualizes:
        For each image analyzed, heatmap of eye-tracking data and corresponding cluster scatterplot for each fixation point
        in a one-by-one fashion via Matplotlib

    Returns:
        ocr_designations (dicts): all of the ocr_designations for each image analyzed. See analyze_data() for more info
        NOTE: uncomment out the print statement directly before plt.show() in analyze_data() function 
        for ocr_designation alongside heatmap/cluster visualization
    '''
    with open("C:/Users/ljdde/Downloads/CVC/test1/experiment_data.json", 'r') as json_file:
        experiment_data = json.load(json_file)

    #this first block is inefficient; for the purposes of giving user an idea of how many images will be analyzed
    #if analysis is only being done by computer the code can easily be simplified into a single FOR loop
    num_analyses=0
    applicable_trials=[]
    for trial,images in experiment_data.items():
        if image_id in images:
            num_analyses+=1
            applicable_trials.append(trial)

    print(f'Number of analyses: {num_analyses}') #to help user know how long execution will take
    for trial in applicable_trials:
            desired_img=experiment_data[trial][image_id]
            analyze_data(desired_img['gaze_data'],base64_to_image(desired_img['edited_image']),desired_img['widget_width'],desired_img['widget_height'],desired_img['left_x'],desired_img['top_y'],desired_img['screensize'],desired_img['image_number'])


def analyze_single_trial(date_time):
    '''
    Goes through a single trial run from a subject and performs the data analysis on each of the 
    image/question pairs given in that trial run. User needs to input start date-time of trial.

    NOTE: similarly to analyze_single_image(), this code will periodically halt while there are still matpltolib
            plots being displayed. To go through the entire code, the user needs to continuously be closing out each 
            window containing a plot.

    Parameters: 
        date_time (str): the string of the date and time of beginning of trial. format can be found in experiment_data.json

    Visualizations:
        For each image analyzed, heatmap of eye-tracking data and corresponding cluster scatterplot for each fixation point
        in a one-by-one fashion via Matplotlib

    Returns:
        ocr_designations (dicts): all of the ocr_designations for each image analyzed. See analyze_data() for more info
    '''
    with open("C:/Users/ljdde/Downloads/CVC/test1/experiment_data.json", 'r') as json_file:
        experiment_data = json.load(json_file)
   
    #prints number of analyses to be performed; 
    #helps user understand how long function will take
    print(f'Number of Analyses: {len(experiment_data[date_time])}') 

    for img_name, desired_img in experiment_data[date_time].items(): #performs data_analysis for each image in trial
        analyze_data(desired_img['gaze_data'],base64_to_image(desired_img['edited_image']),desired_img['widget_width'],desired_img['widget_height'],desired_img['left_x'],desired_img['top_y'],desired_img['screensize'],desired_img['image_number'])


if __name__ == "__main__": 
    pass
    ### ENTER YOUR CODE BELOW TO ANALYZE DATA
    
    ### EXAMPLE: run the following line to analyze data from a single trial run (data initialized in JSON)
    ### analyze_single_trial("2024-07-29 13:44:14")
