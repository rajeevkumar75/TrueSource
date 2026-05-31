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

## AI Text Detection (Human vs AI)

Put labeled samples in `data/text data/all.jsonl` (one JSON object per line):

```json
{"text": "Your passage here...", "label": "human"}
{"text": "Another passage...", "label": "ai"}
```

You can also use folders: `data/text data/human/*.txt` and `data/text data/ai/*.txt`.

Train the text detector:

```bash
pip install -r requirements.txt
python train_text.py --data_path "data/text data/all.jsonl" --epochs 3
```

Run inference on a string or file:

```bash
python predict_text.py --text "Paste your text here"
python predict_text.py --file sample.txt
```

The Streamlit app includes a **Text Detection** tab after training.

Check model quality (test split + fresh holdout samples):

```bash
python scripts/eval_text_model.py
```

## Web App

Full website with Home, Features, About, Contact, and built-in analyzer (image / video / text):

```bash
pip install -r requirements.txt
python run_web.py
```

Open **http://127.0.0.1:5000** — free to use, no sign-in.

## Streamlit Inference Page

Start the Streamlit UI (no login):

```bash
streamlit run app/streamlit_app.py
```