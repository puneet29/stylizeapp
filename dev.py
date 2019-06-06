from bs4 import BeautifulSoup
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session
import os
import requests
from stylize import stylize
import threading
import time
from werkzeug.utils import secure_filename
import logging
import boto3
import botocore


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)

# Set the secret key to some random bytes. Keep this really secret!
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24))


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

            # S3 Bucket
            bucketName = "styletransferbucket"

            filename = secure_filename(file.filename)
            output_img = 'static/out/' + time.ctime().replace(' ', '_')+'.jpg'
            stylize(file, 1, output_img, "models/"+style+".model", 0)

            # S3 upload image
            s3 = boto3.client('s3')
            s3.upload_file(output_img, bucketName, output_img)

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

    # S3 download
    Bucket = "styletransferbucket"
    Key = session['file']
    outPutName = "stylize.jpg"

    s3 = boto3.resource('s3')
    try:
        s3.Bucket(Bucket).download_file(Key, outPutName)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise


if __name__ == "__main__":
    app.run()
