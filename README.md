# Function variable type prediction

## Project Description

The purpose of this project is to build a predictive model that, from the code of a function (and, possibly, the identifier of the project to which it belongs), predicts the types of its different inputs.

## Instructions on how to run the code

In this folder, you can find **app** which is meant to keep the codes for different apps that I have created.
Inside the **app** folder, we can find these files:

- **app_flask.py** , the app that I have created to load the NN model and run the inference. This app is based on the flask package and will be locally loaded on `http://localhost:8080`.
- **app_gradio.py** , the app that I have created to load the NN model and run the inference. This app is based on the gradio package, which is more flexible in terms of design and ML stability and doesn't need a separate HTML to show the results. This will be locally loaded on `http://localhost:8080`.
-**Dockerfile** , which keeps the information for running a docker using the app.
-**requirements.txt** , which keeps the python packages that should be installed before running this app.
-**model** folder, which keeps the model and other encoders that are saved during the exploration phase.
-**templates** folder, which keeps the website HTML that should be used in the **app_flask.py**.



There are two ways to run this app:

- Either using Docker:
	- First, the user should clone the whole folder.
	- Inside the app folder, he will need to run `docker build -t app_gr:v1 .` to build the image and then `docker run -it -p 8080:8080 app_gr:v1` to run the app.
	- An address will be available, and when we go to the address, there is a specific placeholder to put the function body and then see the results.

- Or direct python command: 
	- First, the user should clone the whole folder.
	- Then, he/she will need to go to the app folder.
	- Then, the user should build a new environment using this command `virtualenv -p /usr/bin/python3.8 app_c`.
	- Then, the user should activate the environment like this, `source app_c/bin/activate`.
	- Inside the app folder, he will first run `pip install -r requirements.txt`.
	- Then he can either run `python app_flask.py` or `python app_gradio.py`.
	- Then check the address that will be created to use the app.

- I have also created a public link that I can give you the address by email. Users won't need to do anything here and can check directly the public address.

- Here is an example for the input:
	- function zipobject(props, values) { return basezipobject(props || [], values || [], assignvalue) }

I have also taken into account that the user might put non-relevant information in the input, or if the format is not perfectly aligned with a Java function format.





