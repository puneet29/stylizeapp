from bs4 import BeautifulSoup
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session
import os
import requests
from stylize import stylize
import threading
import time
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = "temp/"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set the secret key to some random bytes. Keep this really secret!
app.secret_key = b'os24/.[lNWfPn<Fs]IuH'


def allowed_file(filename):
    return('.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


def scrape(filename=None):
    preLink = "https://raw.githubusercontent.com/puneet29/StyleTransferApp/master/images/style-images/"
    if(filename == None):
        page = "https://github.com/puneet29/StyleTransferApp/tree/master/images/style-images"
        links = []
        r = requests.get(page)
        soup = BeautifulSoup(r.text, "html.parser")
        tbody = soup.find("tbody")
        photo_anchors = tbody.find_all("a", {"class": "js-navigation-open"})
        for anchors in photo_anchors:
            links.append(preLink+anchors['title'])
        return(links)
    else:
        return(preLink+filename)


@app.route('/')
def homepage():
    if 'file' in session:
        os.remove(session['file'])
        session.clear()
    return render_template('index.html', styles=scrape())


@app.route('/upload/', methods=['POST', 'GET'])
def upload():
    if 'file' in session:
        os.remove(session['file'])
        session.clear()
    style = request.args.get('style')
    if (request.method == 'GET'):
        return render_template('upload.html', stylePath=scrape(style+".jpg"))
    else:
        if ('file' not in request.files):
            flash('No file part')
            return(redirect(request.url))
        file = request.files['file']
        if(file.filename == ''):
            flash('No selected file')
            return(redirect(request.url))
        if(file and allowed_file(file.filename)):
            filename = secure_filename(file.filename)
            output_img = 'static/out/' + time.ctime().replace(' ', '_')+'.jpg'
            uploaded_img = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(uploaded_img)
            stylize(file, 1, output_img, "models/"+style+".model", 0)
            os.remove(uploaded_img)

            while(not os.path.exists(output_img)):
                continue
            session['file'] = output_img
            return(render_template("uploaded.html"))

        flash('Please select right file')
        return(redirect(request.url))


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html')


@app.errorhandler(500)
def server_error(e):
    return render_template('500.html')


@app.errorhandler(405)
def method_error(e):
    return render_template('405.html')


@app.route('/download/')
def download():
    if('file' in session):
        return(send_file(
            session['file'],
            mimetype='image/jpeg',
            attachment_filename='image.jpg',
            as_attachment=True
        ))
    return(render_template('404.html'))


if __name__ == "__main__":
    app.run(debug=True)
