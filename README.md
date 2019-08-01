# Disaster Response Pipeline Project

### Summary:
This projects extracts labelled message data sent to social media during disasters.  The data is cleaned and stored in a database table, which is then used to train a classifier.  The classifier is then used as a back end to a web site, which takes messages as user input and classifies the message.

### Files:
- data/
  - disaster_messages.csv - contains the messages
  - disaster_categories.csv - contains the labels
  - process_data.py - cleans and stores the data
  - DisasterResponse.db - a prepared database
- models/
  - my_transformers.py - contains custom data transformers
  - train_classifier.py - trains the classifier
- app/
  - run.py - launches the web page
  - templates/ - web content


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
