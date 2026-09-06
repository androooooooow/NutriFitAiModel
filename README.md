NutriFit AI Model - Setup Instructions
========================================

This model detects Filipino food items in a photo and looks up their
nutrition information (calories, protein, carbs, fat).

INSTALLATION
------------
1. Create a virtual environment:
   python -m venv venv

2. Activate the virtual environment:
   venv\Scripts\activate

3. Install the required packages:
   pip install ultralytics

After you install all the requirements needed for this model, you need to
change the model path — it depends on what folder you are using, so update
the MODEL_PATH variable in app.py to point to wherever your best.pt file
is actually located on your machine.

RUNNING THE APP
----------------
python app.py
