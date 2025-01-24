Hi Dylan,

I have attached a Python file with a condensed version of my algorithm, let me know if Outlook blocks the attachment.

A few things to note:
The algorithm does not scale the size of the galaxies, only the pixel values.
The derotation portion of the algorithm should be applicable to any images you have, as long as they have been thresholded suitably and do not contain background galaxies.
The thresholding portion of the algorithm has been finetuned for my use case and dataset. I think it should still be applicable to images from other datasets, but you might need to tune some of the parameters to achieve good results for your dataset. (You can also replace it with any other algorithms, but it
I have made a habit of documenting the code, but if anything is unclear or you are uncertain of what it does, please let me know.

Kind regards,
Kevin