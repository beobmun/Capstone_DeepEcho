from flask import Flask, render_template, request, redirect, url_for, send_file, session, jsonify, Response, flash, send_from_directory
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit
import os
import cv2
import threading
import queue
import zipfile

from utils.segmentation import *
from utils.esv_edv import *
from utils.classification import *

app = Flask(__name__)
app.secret_key = 'supersecretkey'
ALLOWED_EXTENSIONS = {'zip'}
VIDEO_EXTENSTIONS = {'mp4', 'avi'}
UPLOAD_FOLDER = 'uploads'
VIDEO_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ESDV_FOLDER = 'esdv'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['ESDV_FOLDER'] = ESDV_FOLDER
socketio = SocketIO(app, cors_allowed_origins="*")

result_queue = queue.Queue()

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
# os.makedirs(ESDV_FOLDER, exist_ok=True)

def convert_to_mp4(file_path):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mp4_file_path = os.path.splitext(file_path)[0] + ".mp4"
    out = cv2.VideoWriter(mp4_file_path, fourcc, fps, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()
    out.release()
    os.remove(file_path)
    return mp4_file_path

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_zip(file_path, extract_to):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def process_classification(video_path, file_path):
    # imgs, video_segment, fps = segment_video(video_path, app.config['PROCESSED_FOLDER'])
    video_dir = classification_a4c(video_path)
    global VIDEO_FOLDER
    VIDEO_FOLDER = video_dir
    os.remove(file_path)
    # result_queue.put((imgs, video_segment, fps))
    socketio.emit('classification_complete', {'status': 'completed'})

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)            
            extract_zip(file_path, app.config['UPLOAD_FOLDER'])
            video_dir = os.path.join(app.config['UPLOAD_FOLDER'], os.path.splitext(filename)[0])
            # global VIDEO_FOLDER
            # VIDEO_FOLDER = classification_a4c(video_dir)
            # VIDEO_FOLDER = video_dir
            
            threading.Thread(target=process_classification, args=(video_dir,file_path,)).start()
            # video_name = os.path.basename(video_path)
            # segmented_path = os.path.join(app.config['PROCESSED_FOLDER'], video_name)
            # session['segmented_filename'] = os.path.basename(segmented_path)
            return redirect(url_for('processing_clf'))
            # os.remove(file_path)
            # return redirect(url_for('videos'))
        
        flash('Invalid file type')
        return redirect(request.url)
    return render_template('upload.html')

@app.route('/processing_clf')
def processing_clf():
    return render_template('processing_clf.html')

def get_video_files(dir_path):
    files = [f for f in os.listdir(dir_path) if f.rsplit('.', 1)[1].lower() in VIDEO_EXTENSTIONS]
    files.sort()
    return files

@app.route('/video/<filename>')
def video(filename):
    return send_from_directory(VIDEO_FOLDER, filename)

@app.route('/videos', methods=['GET', 'POST'])
def videos():
    video_files = get_video_files(VIDEO_FOLDER)
    if not video_files:
        flash("No videos available. Please upload a valid ZIP file containing videos.")
        return redirect(url_for('upload_file'))
    
    initial_video = video_files[0]
    return render_template('video_player.html', videos=video_files, initial_video=initial_video)

def process_segmentation(video_path):
    imgs, video_segment, fps = segment_video(video_path, app.config['PROCESSED_FOLDER'])
    result_queue.put((imgs, video_segment, fps))
    socketio.emit('segmentation_complete', {'status': 'completed'})

@app.route('/segment', methods=['POST'])
def segment():
    # video_path = request.form.get('video_path')
    video_path = request.form['video_path']
    video_path = os.path.join(VIDEO_FOLDER, video_path)
    upload_dir = app.config['UPLOAD_FOLDER']
    os.system(f'cp "{video_path}" "{upload_dir}"')
    session['original_filename'] = os.path.basename(video_path)
    # original_filename = session.get('original_filename')
    # if original_filename:
        # original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        # threading.Thread(target=process_segmentation, args=(original_path,)).start()
    threading.Thread(target=process_segmentation, args=(video_path,)).start()
    video_name = os.path.basename(video_path)
    segmented_path = os.path.join(app.config['PROCESSED_FOLDER'], video_name)
    session['segmented_filename'] = os.path.basename(segmented_path)
    return redirect(url_for('processing_seg'))


@app.route('/processing_seg')
def processing_seg():
    return render_template('processing_seg.html')

@app.route('/result')
def display_result():
    original_filename = session.get('original_filename')
    segmented_filename = session.get('segmented_filename')
    print("##############Original filename: ", original_filename)
    print("##############Segmented filename: ", segmented_filename)
    imgs, video_segment, fps = result_queue.get()
    segmented_path = f"{app.config['PROCESSED_FOLDER']}/{segmented_filename}"
    esdv_path = f"{app.config['ESDV_FOLDER']}/{os.path.splitext(segmented_filename)[0]}"
    graphJSON, hover_images = create_graph(imgs, video_segment, segmented_path, esdv_path, int(fps/4))
    return render_template('result.html', original_filename=original_filename, segmented_filename=segmented_filename, graphJSON=graphJSON, hover_images=hover_images)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename))

if __name__ == '__main__':
    socketio.run(app, debug=True)
