# Import libraries
from fastapi import FastAPI, Form, File, UploadFile
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import shutil
from PIL import Image
# Import classes
from DLImage import DLImage
from DLTextBinary import DLTextBinary
from DLTextMulti import DLTextMulti
from CLTasks import CLTasks

# Initialise FastAPI class
app = FastAPI()

# Task: Can use for form fields, error handling: ValidationError
class DLTextReq(BaseModel):
    model: UploadFile
    text: str
    classes: list
    num_features: int
    num_samples: int = 1000 
    need_process: bool = False
    ds: UploadFile
    ds_text_field: str = None
    max_num: int = None
    max_seq: int = None

class CLReq(BaseModel):
    model: UploadFile
    X_test: UploadFile
    y_test: UploadFile
    selected_all: bool
    selected_graphs: list = None
    target_names: list = None
    sample_indexes: list = None
    scoring: str = 'balanced_accuracy' 
    top_features: int = 5
    feature: str = None
    dist_graph: str = 'boself.sh_expot'
    local_max_features: int = 3
    compacity_nb_features: int = 5
    xai_type: str = 'shap'

# APIs get reports for XAI tasks
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

@app.post("/dltextbin")
def explain_dltextbinary(req: DLTextReq):
    dl_textbin_xai = DLTextBinary(req.model, req.text, req.classes, req.num_features, req.num_samples, 
                                  req.need_process, req.ds, req.ds_text_field, req.max_num, req.max_seq)
    report_name = dl_textbin_xai.generate_report()
    return {'filename': report_name}

@app.post("/dltextmulti")
def explain_dltextmulti(req: DLTextReq):
    dl_textmulti_xai = DLTextMulti(req.model, req.text, req.classes, req.num_features, req.num_samples, 
                                  req.need_process, req.ds, req.ds_text_field, req.max_num, req.max_seq)
    report_name = dl_textmulti_xai.generate_report()
    return {'filename': report_name}

@app.post("/cltask/{task}")
def explain_cltask(req: CLReq, task: str):
    cl_xai = CLTasks(task, req.model, req.X_test, req.y_test, req.selected_all, req.selected_graphs, req.target_names, req.sample_indexes, req.scoring, 
                    req.top_features, req.feature, req.dist_graph, req.local_max_features, req.compacity_nb_features, req.xai_type)
    report_name = cl_xai.generate_report()
    return {'filename': report_name}
