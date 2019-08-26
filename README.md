# Deceit-detection

There is create_data.py script that builds a dataset of 1 Million in csv format with 39 attributes: 
python create_data.py -o <outputFile in csv format> -n <number of rows>


To run facial_exp_deceit_detector.py script use the following format:
python facial_exp_deceit_detector.py -i <training data> -t <test data> -o <output prediction file>

To run hand_gestures_deceit_detector.py script use the following format:
python hand_gestures_deceit_detector.py -i <input file> -o <output file>

Install scikit-plot using:
pip install scikit-plot

Import all necessary libraries:
Pandas,numpy,sklearn,seaborn,matplotlib and scikit-plot

Testing data for facial_exp_deceit_detector.py is saved in All_Gestures_Deceptive and Truthful.csv file

Testing data for hand_gestures_deceit_detector.py is saved in Postures.csv file
