from robyn import Robyn
from transformers import pipeline
import json

app = Robyn(__file__)

# classifier = pipeline("text-classification", device=0)


@app.startup_handler
async def startup_event():
    global inference_handler
    inference_handler = pipeline("text-classification", device=-1)


@app.get("/health")
async def health():
    return "OK"


@app.post("/predict")
async def predict(request):
    body = json.loads(bytearray(request["body"]).decode("utf-8"))
    pred = inference_handler(body["inputs"])
    return json.dumps(pred)


app.start(port=5000)
