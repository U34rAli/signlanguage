# Sign Language Detection

### Steps

### Download Following files and place them into __models__ directory

- YOLOv3 Cross-Dataset
  - [Configuration](https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands.cfg)
  - [Weights](https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands.weights)

### Install Requirements

- pip install -r requirements.txt

### Use Jupyter Notebook

- [Download and past](https://www.kaggle.com/signnteam/asl-sign-language-pictures-minus-j-z/download) your dataset into __data__ folder like this pattern.
  -   data\A\\*
  -   data\B\\*
- Extract features using __feature_extraction.ipynb__ file. It will take some time.
  - Or use this [link](https://srv-file8.gofile.io/downloadStore/srv-store3/n0oZit/abc.zip) (might not work always) to download already featured extracted data. And place into current directory like abc/.
- Feature will be saved into __abc__ folder
- Run __model.ipynb__ file to train and test model
