# Trace Overview
This page provides an overview of the trace of the main_GUI and main_data_analysis programs. Under each file header, a brief high level description is given followed by 
a more detailed description of the program's trace.

Specific lines are not referred to but the trace is described sequentially.

## main_GUI
**High-level**: the user-interface is created using Tkinter, a python-based library that builds GUI softwares. 
The actual tkinter structure (including initialization, frames, widgets, etc) is all located at the bottom of the code, below " if \_\_name\_\_ == '\_\_main\_\_' ".
The functions that are called throughout the tkinter script all lie above this line.
The hyperparameters related to the script are found at the top, just below imported modules.

**Trace**:
To outline the trace, we will establish all of the function bindings on each page of the actual Tkinter GUI, and detail what is executing upon interacting with each of these.

_Welcome page_:

When the app opens, there is a single Start button which, upon clicking:
- calls nextpage(), which executes the following:
    - transitions out of the Welcome page by discarding the welcome widgets
    - calls image_next(), which:
        - fetches an image & question and displays it in the document widget
        - updates the display dimensions left_x and top_y, necessary for data analysis
        - if the desired number of images have already been shown, terminates the experiment and saves the corresponding data to the JSON
- then calls threading(), which:
    - begins a new thread that executes run_eye_tracker(), which:
        - begins running the eye tracker and collecting data until directed to stop
        - after done collecting data, creates a dictionary off all the data for this current image/question pair (curren_question_answer_info), then saves this dictionary to a global dictionary storing data for the entire experiment trial (current_trial_data)

_image/question page_ (recurring):

The only method of leaving this page is pressing the "Enter" button, which, upon clicking:
- calls stop_data_collection(), which:
  - tells run_eye_tracker() to stop collecting data
  - afterwards, calls question_answered() which:
    - displays the intermediary page between image/question pairs.
    - Allows the user to reset and dictate when they are ready for the next question

_intermediary page_ (recurring):

There is a 'Next Question' button at this point, which, upon clicking:
- calls nextpage(), which:
  - same execution as original Start button, except transitions GUI to next image/question from intermediary page instead of welcome page
- calls threading(), which:
  - same as Start button

The _image/question_ and _intermediary_ pages simply alternate until the desired number of images is shown, upon which image_next() will indicate that the experiment is to be terminated.

_Escape Key_:

At the end of (and throughout) the experiment, the user can press the Escape key to quit the app. HOWEVER, if this is done while the eye tracker is running in another thread, 
this thread will continue and the eye tracker will not stop. The code editor, to my knowledge, will have to restart in order to terminate this alternate thread.


## main_data_analysis

**High-level**: this file simply contains many functions, all of which lead up to the primary function analyze_data(). This function is then used in analyze_single_image() and
analyze_single_trial() which are both designed to help streamline analyzing useful data for the user. 

**Trace**:
In order to outline the trace, we will cover each function's use in order to build up to the primary analyze_data() and then continue onto further functions.

- gaussian_2d():
  - calculates gaussian values on a 2D surface.
- heatpoint():
  - used to increment a cartesian coordinate grid radially according to a Gaussian distribution (calls gaussian_2d() to do this).
  - Used in heatmap construction.
- screen_to_image_pixels():
  - converts units of proportion of screen width/height, to the actual pixel number relative to the displayed document image
  - returns None if location is not on image
  - used in ocr_cluster() and analyze_data()
- image_prop_to_image_pixels():
  - converts units of proportion of image width/height to the corresponding pixel coordinates relative to displayed image
  - used in ocr_cluster()
- generate_random_color():
  - generates a random color, used for clustering visualization
- nearest_ocr()
  - finds all the OCRs within a given radius of a gaze location
- ocr_cluster()
  - clusters all the gaze data by time and space, representing all the fixation points
  - calls nearest_ocr() for every single cluster, finding each fixation point's nearest OCRs
  - uses generate_random_color() to designate a color for each cluster
  - returns data for the cluster plot, arrow segments to depict on temporal progress of gaze on heatmap, & each cluster's OCR designations as well
- analyze_data()
  - creates a heatmap overlaying a given document image, then calls heatpoint() on each gaze point to construct the heatmap
  - calls ocr_cluster() to retrieve data regarding tracking data
  - adds temporal progress arrows to heatmap
  - creates scatterplot with corresponding colors for each cluster
  - option to print OCR designations so that they are visible for a human to view alongside plots
  - plt.show() plots and then return a dictionary of the ocr_designations
 
Functions to streamline analysis for a human:
- base64_to_image():
  - for converting Image into base64 string for data storage
- analyze_single_image():
  - prints the number of analyses to be done (so user has an idea of runtime)
  - one by one, shows the matplotlib plots for the analysis of a single image, for every trial it has been a part of
    - Note: as of now, the function only shows one trial at a time. Each window for the plots must be closed before the script continues and shows the next trial. This is room for improvement in the future.
- analyze_single_trial():
  - prints the number of analyses to be done (so user has an idea of runtime)
  - one by one, shows the matplotlib plots for the analysis of a single trial run, including each image shown in the trial
    - Note: same as above

## Recommendations for Future Modifications

Outside of the two notes mentioned in the two functions directly above, here are further recommendations for things to consider in order to improve this framework:

- OCR designation radius:
    - As of now, the nearest_ocr() uses a radius of a constant number of pixels to capture the OCRs around a given fixation point. The default parameter of 72 px was calculated with calculations along with some rough estimations and educated guesses.
    - In reality, the [Perceptual Scan](https://www.researchgate.net/publication/49694262_Eye_movements_the_perceptual_span_and_reading_speed) of a visual fixation point is going to be more elliptical, with left-to-right reading humans perceiving further to the right of the fixation point (about 14-15 letter spaces in 12 point font) than to the left of the fixation point (about 3-4 letter spaces).
    - If there is some way to build a tool that is able to draw such an area that will catch OCRs around a fixation point, that would be a more accurate model.
      
- Minimum cluster size:
    - As of now, ocr_cluster() uses a minimum cluster size of 5 to 'catch' a visual fixation.
    - How this number was calculated:
        - Eye tracker samples at 60 Hz (about 1 every 16.67 ms)
        - Avg reading fixation 200-250ms, often in 125-175ms rnage, can be as low as 75ms
        - Thus, 5 cluster size will catch most fixations
    - Things to consider:
        - Is 5 really the "best" sample size number? How can we best determine this?
        - Can we create a function that calculate size of a specific cluster? Can this lead to a calculation of a relative importance/duration of a fixation point? Perhaps some fixation points got split into multiple clusters, how can we detect this? And subsequently assign more importance to this specific fixation point?
          
- Creating own calibration procedure
    - Creating a specialized calibration could provide several benefits, perhaps:
        - Improving accuracy & ease of integration with GUI
        - Including components related to experiment itself (example: having user read different sized fonts to know how close/far to screen they need to sit for the experiment and/or which images have too small of font, etc)
          
- More robust testing for accuracy:
    - Though I am confident in the accuracy of the heatmap and OCR designations, and some basic testing for this was performed, the program could benefit from additional more rigorous testing to ensure its accuracy
    - Particularly for edge cases that may be overlooked in the code (what if user looks at edge of image/screen, away from screen, etc).
