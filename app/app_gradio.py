import pickle
import re
import numpy as np
from tensorflow.keras.models import model_from_json
import gradio as gr

globals()["vectorizer"] = None
globals()["label_encoder_types"] = None
globals()["variable_binarizer"] = None
globals()["model"] = None


# cleaning the body of each function
re_white = re.compile(r"(\s)+", re.UNICODE)


def strip_multiple_whitespaces(s):
    return re_white.sub(" ", s)


def clean_text(s):
    s = s.lower()
    s = s.replace("\n", " ")
    s = s.replace("\t", " ")
    s = s.replace(".", " ")
    s = s.replace(";", " ")
    s = re.sub(r"[^A-Za-zèéêëöôòóœàáâçčćûüùúūîïíīįì,=><)()}{}[]+-]", " ", s)
    s = strip_multiple_whitespaces(s)
    return s


# find the input variables of each function
def variables(s):
    result = s[s.find("(") + 1 : s.find(")")]
    return result


# build the inference to merge the results for each input type
def inference(function):
    if "(" not in function:
        return "No correct input"
    if ")" not in function:
        return "No correct input"
    variables_ = variables(function)
    variables_split = variables_.split()
    if variables_split == []:
        return "Variable not found"
    variables_list = variables_.split(",")
    variables_list = [i.strip() for i in variables_list]
    function = clean_text(function)
    X_test_deep_1 = globals()["vectorizer"].transform([function])
    X_test_deep_1.sort_indices()
    output = []
    for variable in variables_list:
        if variable.split() == []:
            prediction = "Variable is not correct"
            output.append(prediction)
            continue
        X_test_deep_2 = globals()["variable_binarizer"].transform([variable])
        y_pred = globals()["model"].predict(
            [X_test_deep_1, X_test_deep_2], batch_size=1, verbose=0
        )
        y_pred_bool = np.argmax(y_pred, axis=1)
        prediction = globals()["label_encoder_types"].inverse_transform(y_pred_bool)
        output.append(prediction[0])
    output = (",").join(output)
    return output


def load_model():
    # Load label_encoder_types, vectorizer and variable_binarizer
    globals()["label_encoder_types"] = pickle.load(
        open("model/label_encoder_types.sav", "rb")
    )
    globals()["vectorizer"] = pickle.load(open("model/vectorizer.sav", "rb"))
    globals()["variable_binarizer"] = pickle.load(
        open("model/variable_binarizer.sav", "rb")
    )

    # load json and create model
    json_file = open("model/model_NN.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    globals()["model"] = model_from_json(loaded_model_json)
    # load weights into new model
    globals()["model"].load_weights("model/model_NN.tf")


def predict(function):

    if function == "":
        prediction = "No input"
    elif "function" not in function:
        prediction = "No correct input"
    else:
        prediction = inference(function)
    return prediction


# Running the app
if __name__ == "__main__":
    load_model()
    print("======= Model loaded =======")
    demo = gr.Interface(
        title="Predicting the type of variables of a function",
        fn=predict,
        inputs=gr.Textbox(lines=2, placeholder="Paste function here ..."),
        outputs="text",
        allow_flagging="never",
        description="This app predicts the type of variables of a function. You should paste the function body into the textbox and press the submit button.",
    )
    demo.launch(server_port=8080, server_name="0.0.0.0")
