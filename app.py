from flask import Flask, request, render_template
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import os
import fitz  
import torch.nn.functional as F
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


model_path = "./model"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path, problem_type="multi_label_classification")
model.eval()

label_map = {
    0: 'Advocate', 
    1: 'Arts', 
    2: 'Automation Testing', 
    3: 'Backend Developer', 
    4: 'Blockchain', 
    5: 'Business Analyst', 
    6: 'Civil Engineer', 
    7: 'Cloud Engineer', 
    8: 'Data Scientist', 
    9: 'Database', 
    10: 'DevOps Engineer', 
    11: 'DotNet Developer', 
    12: 'ETL Developer', 
    13: 'Electrical Engineering', 
    14: 'Frontend Developer', 
    15: 'Full Stack Developer', 
    16: 'HR', 
    17: 'Hadoop', 
    18: 'Health and fitness', 
    19: 'Java Developer', 
    20: 'Machine Learning Engineer', 
    21: 'Mechanical Engineer', 
    22: 'Mobile App Developer (iOS/Android)', 
    23: 'Network Security Engineer', 
    24: 'Operations Manager',
    25: 'PMO', 
    26: 'Python Developer',
    27: 'SAP Developer', 
    28: 'Sales', 
    29: 'Testing', 
    30: 'Web Designing'
}

def read_file(file):
    if file.filename.endswith('.pdf'):
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            text = " ".join([page.get_text() for page in doc])
        return text
    else:
        return file.read().decode('utf-8')

def predict_top_k(resume_text, k=3):
    inputs = tokenizer(resume_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    top_probs, top_indices = torch.topk(probs, k=10, dim=1)  

    predictions = []
    for i in range(top_indices.shape[1]):
        label_idx = top_indices[0][i].item()
        label = label_map.get(label_idx, f"Unknown ({label_idx})")
        score = top_probs[0][i].item()
        if not label.startswith("Unknown"):
            predictions.append((label, score))
        if len(predictions) == k:
            break  

    return predictions


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', label_map=label_map, predictions=None)
    predictions = []
    if request.method == 'POST':
        role = request.form.get("target_role")
        target_label = {v: k for k, v in label_map.items()}.get(role)
        files = request.files.getlist("resume_files")
        if not any(f.filename for f in files):  
            return render_template('index.html', label_map=label_map, predictions=None, message="⚠️ No files uploaded.")
        for f in files:
            text = read_file(f)
            top_k_preds = predict_top_k(text, k=3)  
            
            
            is_fit = any(pred[0] == role for pred in top_k_preds)

            predictions.append({
                "filename": f.filename,
                "top_k_predictions": top_k_preds,
                "is_fit": is_fit
            })
    
    return render_template('result.html', label_map=label_map, predictions=predictions)




if __name__ == '__main__':
    app.run(debug=True)
