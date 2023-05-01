from flask import Flask , render_template , request ,jsonify
from transformers import AutoModel, AutoTokenizer,pipeline

app = Flask(__name__) 

tokenizer = AutoTokenizer.from_pretrained("./model/tf_model.h5")
model = AutoModel.from_pretrained("./model/tf_model.h5")
emotion=pipeline('sentiment-analysis',model=model)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit',methods=['POST'])
def submit():
    text = request.form['text-input']
    emotion_p=emotion(text)
    return jsonify(emotion_p)

if __name__ == '__main__':
    app.run(host='0.0.0.0')

    
           