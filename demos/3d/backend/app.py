from flask import Flask, request
from flask_cors import CORS
import torch
from werkzeug.utils import secure_filename
import os
import pandas as pd
import json
from SELDNet import Seldnet_augmented 
import utils
import librosa

app = Flask(__name__)

# Allow 
CORS(app)

# Path for uploaded images
UPLOAD_FOLDER = 'uploads/'
MODEL_PATH = 'model/seldnet_checkpoint'

# Allowed file extransions
ALLOWED_EXTENSIONS = set(['wav', 'csv'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def hello():
	return "Hello World!"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_wav', methods=['GET', 'POST'])
def upload_audio_file():
	if request.method == 'POST':
		print("request data", request.data)
		print("request files", request.files)

		# check if the post request has the file part
		if 'file' not in request.files:
			return 'ERROR'
		wav_file = request.files['file']

		if wav_file and allowed_file(wav_file.filename):
			# predictions
			wav_filename = secure_filename(wav_file.filename)
			wav_file.save(os.path.join(app.config['UPLOAD_FOLDER'], wav_filename))
			return json.dumps(get_predictions(UPLOAD_FOLDER+wav_filename))

	return 'ERROR'

@app.route('/upload_csv', methods=['GET', 'POST'])
def upload_ground_truth_file():
	if request.method == 'POST':
		print("request data", request.data)
		print("request files", request.files)

		# check if the post request has the file part
		if 'file' not in request.files:
			return 'ERROR'
		csv_file = request.files['file']

		if csv_file and  allowed_file(csv_file.filename):
			
			# ground truth
			csv_filename = secure_filename(csv_file.filename)
			csv_file.save(os.path.join(app.config['UPLOAD_FOLDER'], csv_filename))
			
			return json.dumps(get_ground_truth(UPLOAD_FOLDER + csv_filename))

	return 'ERROR'

def get_ground_truth(file_path):
	# opening the CSV file
	df = pd.DataFrame(pd.read_csv(file_path))
	gt_values = df.values.tolist()
	gt = [
		{'frame': e[1], 'class': e[1] , 'x': str(e[2]), 'y': str(e[3]), 'z': str(e[4])} for e in gt_values
	]

	return gt
	

def get_predictions(audio_path):
	model = Seldnet_augmented(
		time_dim=2400,
		freq_dim=256,
		input_channels=4,
		output_classes=14,
		pool_size=[[8, 2], [8, 2], [2, 2], [1, 1]],
		pool_time=True,
		rnn_size=256,
		n_rnn=3,
		fc_size=1024,
		dropout_perc=0.3,
		cnn_filters=[64, 128, 256, 512],
		class_overlaps=3,
		verbose=False,
	)	

	model = model.to('cpu')
	_ = utils.load_model(model.to('cpu'), None, MODEL_PATH, False)
	model.eval()

	x, sr = librosa.load(audio_path, sr=32000, mono=False)
	x = utils.spectrum_fast(
		x, nperseg=512, noverlap=112, window='hamming', output_phase=False
	)

	x = torch.tensor(x).float().unsqueeze(0)
	with torch.no_grad():
		sed, doa = model(x)
	sed = sed.cpu().numpy().squeeze()
	doa = doa.cpu().numpy().squeeze()
	
	pred_list = utils.predictions_list(sed, doa)
	pred_dict = [
		{'frame': e[1], 'class': e[1] , 'x': str(e[2]), 'y': str(e[3]), 'z': str(e[4])} for e in pred_list
	]

	return pred_dict


if __name__ == "__main__":
	app.run(debug=True)