Mars surface image (Curiosity rover) labeled data set
-----------------------------------------------------
Authors: Alice Stanboli and Kiri L. Wagstaff
Contact: kiri.l.wagstaff@jpl.nasa.gov

This data set consists of 6691 images that were collected by the Mars
Science Laboratory (MSL, Curosity) rover by three instruments (Mastcam
Right eye, Mastcam Left eye, and MAHLI).  These images are the
"browse" version of each original data product, not full resolution.
They are roughly 256x256 pixels each.  Full-size images can be
obtained from the PDS at https://pds-imaging.jpl.nasa.gov/search/ .

We divided the MSL images into train, validation, and test data sets
according to their sol (Martian day) of acquisition.  This strategy
was chosen to model how the system will be used operationally with an
image archive that grows over time.  The images were collected from
sols 3 to 1060 (August 2012 to July 2015).  The exact
train/validation/test splits are given in individual files.

Contents:
- calibrated/: Directory containing calibrated MSL browse images
- train-calibrated-shuffled.txt: Training labels (images in shuffled order)
- val-calibrated-shuffled.txt: Validation labels
- test-calibrated-shuffled.txt: Test labels
- msl_synset_words-indexed.txt: Mapping from class IDs to class names

Attribution:
If you use this data set in your own work, please cite this DOI:

10.5281/zenodo.1049137

Please also cite this paper, which provides additional details about
the data set.

Kiri L. Wagstaff, You Lu, Alice Stanboli, Kevin Grimes, Thamme Gowda,
and Jordan Padams. "Deep Mars: CNN Classification of Mars Imagery for
the PDS Imaging Atlas." Proceedings of the Thirtieth Annual Conference
on Innovative Applications of Artificial Intelligence, 2018.

