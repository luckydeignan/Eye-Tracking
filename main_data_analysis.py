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
    Converts screen proportion coordinates to pixel value coordinates, 
    primarily used in heatmap construction

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
            pass
    except: # usually occurs when a cluster is located to be off the screen; 
            # likely a result of faulty calibration of the eye tracker
        raise(Exception)
    

def screen_to_image_coords(coords,display_size,reduced_image_size,widget_height,widget_width,left_x,top_y):
    '''
    Converts screen proportion coordinates to image proportion coordinates,
    used to help with nearest_ocr detection

    Parameters: 
    coords (tuple): gaze coordinate relative to display (i.e (x,y))
    display_size (tuple): size of display in pixels (i.e (width,height))
    reduced_image_size (tuple): size of displayed image in pixels (i.e (width,height))
    widget_height, widget_width, left_x, top_y (ints): pixel values that bound the image widget and describe leftward and upward portion of screen

    Returns 
    image_proportions (tuple): proportinal coords (x,y) of gaze coordinate relative to image 
                                ONLY if gaze location is on displayed image. Else, return None
    '''
    x = coords[0]
    y = coords[1]
    width = reduced_image_size[0]
    height = reduced_image_size[1]
    screen_width, screen_height = display_size[0],display_size[1]

    #pixel values that bound the document image
    x_max = left_x + widget_width - (widget_width-width)//2
    y_max = top_y + widget_height - (widget_height-height)//2
    x_min = left_x + (widget_width-width)//2
    y_min = top_y + (widget_height-height)//2

    image_pixels = screen_to_image_pixels(x,y,x_min,x_max,y_min,y_max,screen_width,screen_height)
    if image_pixels:
        image_proportions = image_pixels[0]/width , image_pixels[1]/height

        return image_proportions
    else:
        return None,None

def nearest_ocr(ocr_info,gaze_location,rad=.5):
    '''
    Parameters: 
    ocr_info (list of dicts): this is gathered from imdb data. Each dict is an OCR in the image, containing its location
    avg_gaze_location (tuple): the avg coordinates of the cluster being inquired. Units are proportion of file
    rad (float): radial distance, in units of screen proportion, that indicates the region of OCRs to be captured for a given gaze point
    NOTE: due to difference in height/width of display, this will not result in a perfect circle around a gaze point

    Returns: 
    (str): nearest OCR tokens as a string
    '''
    ocr_prep = []
    for ocr in ocr_info: 
        #as of now we are calculating avg ocr location and using that to calculate nearest OCRs
        #this could be played with; could potentially use any edge of box to calculate nearest OCRs
        avg_x = ocr['bounding_box']['topLeftX']+(ocr['bounding_box']['width']/2)
        avg_y = ocr['bounding_box']['topLeftY']+(ocr['bounding_box']['height']/2)
        ocr_prep.append((avg_x,avg_y))

    ocr_locations = np.array(ocr_prep)

    tree = KDTree(ocr_locations) #using KDTree for more efficient searching

    #NOTE: rad is important parameter. Determines the radius around fixation point that catches all the OCRs. 
    # Empirically, we've found .5 so far to be a reasonable number for this parameter
    # However, the units are in proportion of screen width (x-variable) and height (y-variable)
    # since a display is often rectangular, this results in an ellipse rather than circle around the gaze point
    # this may want to be revisited in the future if it is really the best method of OCR designation given a gaze coordinate

    indeces = tree.query_ball_point(gaze_location,rad)

    nearest_tokens = [ocr_info[index] for index in indeces]

    return([token['word'] for token in nearest_tokens])



def ocr_cluster(eye_data,image_number,screensize,reduced_width,reduced_height,widget_width,widget_height,left_x,top_y,min_cluster_size=5):
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
    # Eye-tracking data samples at 60Hz, meaning 1 every 16.67 seconds
    # Avg reading fixation duration 200-250 ms, often in 125-175 ms range, can be as low as 75 ms
    # 
    # Given these numbers, min_cluster_size of 5 will catch even the smallest of fixations, but avg cluster size should be ~12-15
    # This logic and implementation can be tested and played around with for further adjustment as needed
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size)


    labels = clusterer.fit_predict(standardized_data) #get arbitrary cluster labels


    # color functions for cluster visualization purposes

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

    def generate_random_color(threshold=0.2):
        '''Generate a random hexadecimal color that is not too close to white or black.'''
        while True:
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            lightness = calculate_lightness(r, g, b)
            
            # Ensure the lightness is not too close to black (0) or white (1)
            if threshold < lightness < (1 - threshold):
                return rgb_to_hex(r, g, b)

    
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


  
    ocr_data = np.load("C:/Users/ljdde/Downloads/CVC/spdocvqa_imdb/imdb_train.npy",allow_pickle=True) #load in ocr data

    ocr_info = ocr_data[image_number]['ocr_info'] #ocr info for this speciifc image

    ocr_designation = {} # going to have each color cluster with corresponding OCRs
 
    #this next block calculates nearest OCRs for each cluster

    for cluster in cluster_info:
        #convert display proportion coordinates to image proportion coordinates
        final_x,final_y = screen_to_image_coords((cluster[1][1],cluster[1][2]),screensize,(reduced_width,reduced_height),widget_height,widget_width,left_x,top_y)
            
        #if cluster lies on screen, find nearest OCR and add to ocr_designation
        if final_x and final_y:
            near_ocr = nearest_ocr(ocr_info,(final_x,final_y))
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
        runs the eye-tracker (for hard-coded time for now)
        overlays the heatmap of eye gaze locations onto corresponding image shown in tkinter
        shows the cluster graph of eye-tracking data
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


    #create second plot; cluster graph
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

    ax[1]  = fig.add_subplot(122, projection='3d')
    ax[1].scatter(x, y, z, c=list_colors)


    # Set labels
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('x-location')
    ax[1].set_zlabel('y-location (inverted)')

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

if __name__ == "__main__": 
    with open("C:/Users/ljdde/Downloads/CVC/test1/experiment_data.json", 'r') as json_file:
        experiment_data = json.load(json_file)
    
    desired_img = experiment_data["2024-07-25 11:41:25"]['ffbx0227_7']
    
    print(analyze_data(desired_img['gaze_data'],base64_to_image(desired_img['edited_image']),desired_img['widget_width'],desired_img['widget_height'],desired_img['left_x'],desired_img['top_y'],desired_img['screensize'],desired_img['image_number']))


