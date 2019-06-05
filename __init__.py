from flask import Flask, render_template
from bs4 import BeautifulSoup
import requests
from stylize import stylize

app = Flask(__name__)


def scrape():
    links = []
    page = "https://github.com/puneet29/StyleTransferApp/tree/master/images/style-images"
    r = requests.get(page)
    soup = BeautifulSoup(r.text, "html.parser")
    tbody = soup.find("tbody")
    photo_anchors = tbody.find_all("a", {"class": "js-navigation-open"})
    for anchors in photo_anchors:
        links.append(
            "https://raw.githubusercontent.com/puneet29/StyleTransferApp/master/images/style-images/"+anchors['title'])
    return(links)


@app.route('/')
def homepage():
    return render_template('index.html', styles=scrape())


@app.route('/upload/')
def upload():
    return render_template('upload.html')


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html')


@app.errorhandler(500)
def server_error(e):
    return render_template('500.html')


if __name__ == "__main__":
    app.run(debug=True)
