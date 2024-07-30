# Recommendations for Future Modifications

Outside of the two "Notes" mentioned in analyze_single_iamge() and analyze_single_trial() and the bottom of TraceOverview.md (read that document first, to have an understand of the program before coming here), here are further recommendations for things to consider in order to improve this framework:

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
    - Though I am confident in the accuracy of the heatmap and OCR designations, and some basic testing for this was performed, the program can always benefit from continuous and more rigorous testing to ensure its accuracy, especially as development continues.
    - Particularly for edge cases that may be overlooked in the code (what if user looks at edge of image/screen, away from screen, etc).

- Miscellaneous Small Fixes (small fixes I just didn't have time to get to)
    - Displaying the question/answer alongside the heatmap
        - Would be helpful to know so that we can see if the logic behind the data makes sense
        - When we call analyze_data(), we already have the index of imdb_data associated with the image (image_number), so this should be an easy task
    - Distinguishing off-image gazes and non-OCR gazes
        - As of now, when printing the OCR designations, I believe it prints None for off-image gazes and [] for non-OCR capturing gazes.
        - This needs confirmation though. If so, make it more clear that the subject is gazing away from the image at that moment. 
    - Fixing the last page transition
        - As of now, when the experiment is done, the GUI displays the intermediary page one more time before going to the final page
        - Would be a more coherent and smooth transition to go directly to the end page once experiment is done
