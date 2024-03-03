# PPE-kit-detector
**Overview**
This project aims to enhance the safety of workers at construction sites by implementing a Personal Protective Equipment (PPE) Kit Detector using Python and machine learning models. 
The detector is built upon the YOLO (You Only Look Once) v8 model and has been trained using datasets downloaded from the Roboflow website.

**How it Works**

_Input Images/Video Feed:_ The system takes input from images or a live video feed captured at the construction site.
_Detection Algorithm:_ The YOLO v8 model processes the input, swiftly identifying individuals and detecting whether they are equipped with the required PPE.

_Visualization:_ The areas of interest (individuals without proper PPE) are highlighted in the output, providing a clear visual indication.

**Implementation Details**

_Python Programming:_ The project is implemented in Python, making it versatile and easy to integrate with existing systems.

_Configuration Flexibility_: The config.yml file allows users to tweak parameters such as confidence thresholds, adapting the detector to specific project requirements.
**Usage**
Users can run the PPE Kit Detector with a simple command. The system then processes the input, providing immediate feedback on the safety compliance of individuals at the construction site.
