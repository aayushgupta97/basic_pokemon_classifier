from fastai.vision import *
from fastai.widgets import *
from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse, RedirectResponse
import uvicorn
from io import BytesIO
import aiohttp
app = Starlette(debug=True)

learner = load_learner('')


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    img = open_image(BytesIO(bytes))
    _,_,losses = learner.predict(img)
    
    
    return JSONResponse({
        "predictions": sorted(
            zip(learner.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        )
    })

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
