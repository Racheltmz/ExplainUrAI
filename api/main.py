# Import classes
from deep_learning_image import DLImage
from deep_learning_text_bin import DLTextBinary
from deep_learning_text_multi import DLTextMulti
from classical_learning import CLTasks

# Import libraries
from fastapi import FastAPI, UploadFile
from tensorflow.keras.models import load_model
import shutil
from PIL import Image

# Initialise FastAPI class
app = FastAPI()

# APIs get reports for XAI tasks
# Deep Learning Image Classification
@app.post("/dlimage")
def explain_dlimage(model: UploadFile, imgs: list[UploadFile]):
    # Save the uploaded file
    with open(f'./models/{model.filename}', 'wb') as buffer:
        shutil.copyfileobj(model.file, buffer)

    # Load the model from the h5 file
    img_model = load_model(f'./models/{model.filename}')

    # Save the uploaded images
    for img in imgs:
        with open(f'./images/{img.filename}', 'wb') as buffer:
            shutil.copyfileobj(img.file, buffer)
    dl_imgs = [Image.open(f'./images/{img.filename}') for img in imgs]

    # Get explanation
    dl_img_xai = DLImage(img_model, dl_imgs)
    report_name = dl_img_xai.generate_report()
    return {'filename': report_name}

# Deep Learning Binary Text Classification
@app.post("/dltextbin")
def explain_dltextbinary(
    model: UploadFile, 
    text: str,
    classes: list,
    num_features: int,
    num_samples: int = 1000,
    need_process: bool = False,
    ds: UploadFile = None,
    ds_text_field: str = None,
    max_num: int = None,
    max_seq: int = None):
    dl_textbin_xai = DLTextBinary(model, text, classes, num_features, num_samples, 
                                  need_process, ds, ds_text_field, max_num, max_seq)
    report_name = dl_textbin_xai.generate_report()
    return {'filename': report_name}

# Deep Learning Multi Text Classification (multi-class/multi-label)
@app.post("/dltextmulti")
def explain_dltextmulti(
    model: UploadFile, 
    text: str,
    classes: list,
    num_features: int,
    num_samples: int = 1000,
    need_process: bool = False,
    ds: UploadFile = None,
    ds_text_field: str = None,
    max_num: int = None,
    max_seq: int = None):
    dl_textmulti_xai = DLTextMulti(model, text, classes, num_features, num_samples, 
                                  need_process, ds, ds_text_field, max_num, max_seq)
    report_name = dl_textmulti_xai.generate_report()
    return {'filename': report_name}

# Classical Learning tasks
@app.post("/cltask/{task}")
def explain_cltask(
    task: str, 
    model: UploadFile, 
    X_test: UploadFile, 
    y_test: UploadFile,
    selected_all: bool,
    selected_graphs: list = None,
    target_names: list = None,
    sample_indexes: list = None,
    scoring: str = 'balanced_accuracy' ,
    top_features: int = 5,
    feature: str = None,
    dist_graph: str = 'boself.sh_expot',
    local_max_features: int = 3,
    compacity_nb_features: int = 5,
    xai_type: str = 'shap'):
    cl_xai = CLTasks(task, model, X_test, y_test, selected_all, selected_graphs, target_names, sample_indexes, scoring, 
                    top_features, feature, dist_graph, local_max_features, compacity_nb_features, xai_type)
    report_name = cl_xai.generate_report()
    return {'filename': report_name}
