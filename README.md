# TrueSource: Multi-Model Deepfake Detection

## Image Detection Training

Expected dataset structure:

data/deepfake_images/
  train/real, train/fake
  val/real, val/fake
  test/real, test/fake

Install dependencies:

pip install -r requirements.txt

Run training:

python main.py --data_dir data/deepfake_images --epochs 10 --batch_size 32

## Streamlit Inference Page

Start the test webpage:

streamlit run app/streamlit_app.py