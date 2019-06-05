from flask import Flask, render_template, request, redirect, url_for, flash
from bs4 import BeautifulSoup
import requests
from stylize import stylize
from werkzeug.utils import secure_filename
import os


UPLOAD_FOLDER = "uploads/"
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
    return render_template('index.html', styles=scrape())


@app.route('/upload/', methods=['POST', 'GET'])
def upload():
    if (request.method == 'GET'):
        style = request.args.get('style')
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
            print(file)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return(redirect(url_for('uploaded_file', filename=filename)))
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


@app.route('/uploaded_file/')
def uploaded_file():
    return("Done")


if __name__ == "__main__":
    app.run(debug=True)
